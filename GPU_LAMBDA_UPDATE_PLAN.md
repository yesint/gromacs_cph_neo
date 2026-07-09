# GPU λ-dynamics plan (variant B): proper GPU-resident constant-pH

Branch: `gpu-lambda-update`. Status: **plan / analysis only — nothing implemented.**

This is the *general-purpose* GPU plan, superseding the narrow "classic NB-offload" sketch in
`GPU_PORT_PLAN.md`. Design target is **not** this project's small Martini-RF system but arbitrary
GROMACS constant-pH runs, up to **large PME systems on multiple GPUs**. The goal is that
`mdrun -update gpu -nb gpu -pme gpu` works with `lambda_dynamics` and stays **device-resident**:
x/v/f never leave the GPU between neighbour-search steps, and the λ loop rides along on the device.

The distinguishing requirement vs `GPU_PORT_PLAN.md`: that plan deliberately ran the *classic*
GPU path (buffer-ops OFF, CPU force reduction, per-step full x+q H2D and potential D2H) and
**forbade `-update gpu`**. That is fine for a 46k-bead RF box but throws away GPU-residency — the
whole point of modern GPU GROMACS — and it silently omits PME, so it is wrong for any large system.

---

## 0. What "proper" requires — the residency constraint

The per-step cph data dependency (today, all host — `md.cpp:1249-1329`):

```
setLambdaCharges(mdatoms.chargeA)      # λ → per-atom charges           [O(Natoms) write]
do_force  → NB kernel + PME fill V_i   # per-atom electrostatic potential
reduceElectrostaticPotential → host    # V_i (nbat order) → fr.electrostaticPotential  [O(Natoms)]
updateLambdas(step)                    # V_i → groupPotential → integrate λ (host)
                                       #   → new λ feeds next step's charges
```

There are exactly **two O(Natoms) operations** on this critical path each step: (1) expanding λ into
per-atom charges, and (2) consuming the per-atom potential. GPU-resident update keeps x/v/f on the
device and only D2H-syncs at NS/output steps; if *either* of those two O(Natoms) operations runs on
the host, we pay one full-system PCIe transfer every step and residency is lost. So "proper" means:

> **Both O(Natoms) operations must run on the device.** Everything between them that is O(Ngroups)
> (the λ ODE integration itself) is tiny and where it runs is a cost/complexity trade-off, not a
> residency question.

This yields the central design choice below.

---

## 1. Central design decision — the **hybrid** loop (recommended)

Split the cph step by data size, not by "CPU vs GPU":

- **On device (O(Natoms), must be resident):**
  - `V_i` accumulation in the NB kernel (real space) **and** PME gather (reciprocal space).
  - **potential → groupPotential** reduction: segmented sum of `V_i · Δq_i` over each group's atom
    list (device index buffers), producing one scalar per titratable group.
  - **setLambdaCharges**: expand updated λ → per-atom charges, written into the resident NB `xq` and
    the PME charge buffer, in the correct atom orderings.
- **On host (O(Ngroups), negligible — reuse existing validated code):**
  - the λ ODE step: `computeForces` (pKa / double-well / reference / pH terms), the **collective
    v-rescale thermostat** (global Ekin + shared RNG draw), and the **charge / multi-state
    constraints** solve. This is `updateLambdas` (`constant_ph.cpp:1478`) essentially unchanged.

Each step crosses PCIe with only two **tiny contiguous vectors**: `groupPotential` D2H and the updated
λ (x) H2D — both O(Ngroups), independent of box size. Residency holds for arbitrarily large systems.

**Why hybrid over full-device λ integration.** Porting the λ ODE to the device buys almost nothing
(Ngroups ≪ Natoms) but is where the hardest correctness risks live: reproducing the ThreeFry v-rescale
RNG stream bit-for-bit on device, and the coupled multi-state/charge-constraint solve. Leaving those on
the host keeps the numerically delicate, already-validated code exactly as the CPU port / fork has it,
and confines device work to two conceptually simple data-parallel kernels (a segmented reduction and a
scatter). Full-device λ integration is documented as an **optional** later optimization (L4), justified
only if the O(Ngroups) host hop ever measures as non-negligible (e.g. 10⁴+ titratable groups).

The hybrid is genuinely GPU-resident: `-update gpu` runs, and the only host touch per step is a
kilobyte-scale vector round trip that overlaps trivially.

---

## 2. Foundational gap that must close first — the full potential (CPU, then GPU)

The current port computes `V_i` **only from the real-space NB kernel** (grep: `electrostaticPotential`
is filled solely by `nbnxm/*`). It has **no reciprocal-space (PME) contribution** — the 2026 tree lacks
the fork's `gather_potential` entirely (`ewald/` grep empty). This is correct for reaction-field Martini
(no reciprocal term) but **wrong for any PME system**: dV/dλ would be missing the long-range Coulomb
derivative. No GPU work matters until this is fixed, because there would be nothing correct to make
resident.

The fork **already implements the full path** and is the porting oracle:
- CPU: `ewald/pme_gather.cpp:370 gather_potential_bsplines` → per-atom reciprocal potential.
- GPU: `ewald/pme_gpu.cpp:328` accumulates `output.potentials_` into `electrostaticPotential`
  (host reduce of a device-gathered per-atom potential).
- Assembly: fork `mdlib/sim_util.cpp:1723-1727` adds NB potential; PME potential folded in via the
  PME reduce (`pme_gpu_reduce_outputs`, `pme_gather` on CPU).

Note the fork itself is still a **classic-path** implementation: it D2H-reduces potentials and
integrates λ on the host every step. So even Layer 1 below (potential correct on GPU, λ on host) is a
faithful fork port; only Layer 3 (residency) is new engineering.

---

## 3. Work packages (layered; each layer ends in a validation gate)

Effort is rough dev-days for the CPU-port author now on GPU. **CUDA first** (cluster = `+cuda`); no
local GPU, so all GPU build/test is on aurum2.

### Layer 0 — general-purpose CPU foundation (no GPU; makes the port correct at scale)
- **L0.1 PME per-atom potential, CPU — ✅ DONE (commit 1b90be2), M0a PASS.** Ported
  `gatherPmePotential` into the CPU PME gather so `fr->electrostaticPotential` gets the reciprocal
  term; NB+PME assembled by clearing the buffer before do_force and letting both `+=`. Key subtlety
  found: titratable atoms neutral at the current λ (charge 0, non-zero charge-difference) were skipped
  by `make_bsplines` → forced all-atom splines when gathering (`bDoSplines |= !potentials.empty()`).
  **M0a:** all 46 λ groups' dV/dλ match the 2021 fork to the single-precision floor (max rel 1.1e-5;
  RF regression unchanged at 5e-5). Repro: `full_size/cph/m0a_repro.sh`. Single PME rank only (PME
  decomposition asserts out — deferred to the multi-rank work). See PORT_LOG.md for full detail.
- **L0.2 Multi-rank DD, CPU — 🚧 IN PROGRESS (two bugs; part 1 done).** Turned out to be *two*
  independent DD bugs (see PORT_LOG.md):
  - **✅ part 1 (commit 34dc38b):** the cross-domain group-potential all-reduce + λ-broadcast were dead
    because `ConstantPH` had a `nullptr` commrec. Added `setCommrec()`, injected in runner. No-op at
    single rank. `updateAfterPartition`'s ga2la mapping was already correct.
  - **🚧 part 2 (open):** the per-atom potential on *halo* atoms is not communicated back to home ranks
    (the port reduces only `Local` and has no potential halo move; the fork rides `dd_move_f` by storing
    φ in the 4th force component). Preferred fix: reduce over `All` + an explicit scalar DD halo move,
    keeping the separate potential buffer (aligns with the GPU design). Needs a boundary-straddling
    multi-domain test.
  **Gate M0b:** a 2–4 rank DD cph run reproduces the single-rank λ trajectory (blocked on part 2). Note:
  DD+PME additionally needs the deferred PME-decomposition/separate-PME-rank potential path (§4).

Layer 0 alone upgrades the port from "RF single-rank" to "general CPU cph." It is a prerequisite for
*any* large-system use and is independent of the GPU decision.

### Layer 1 — correct potential on the GPU (classic path, λ still on host)
- **L1.1 NB real-space potential on GPU (3–5d, per-backend kernel).** = `GPU_PORT_PLAN.md`
  WP-G1/G2/G3: add `DeviceBuffer<float> potential` to `NBAtomDataGpu`, clear each step with `f`,
  accumulate per pair in `nbnxm_cuda_kernel.cuh` (`atomicAdd`, or the register-accumulator variant),
  D2H every step (cph-gated), reduce nbat→atom via `gridSet.cells()`, RF/Ewald self-term host-side.
- **L1.2 PME reciprocal potential on GPU (3–4d).** Port the fork's GPU PME potential gather
  (`pme_gpu.cpp` + the PME gather kernel producing `output.potentials_`) to 2026's PME. Combine with
  L1.1 into one `fr->electrostaticPotential`.
- **L1.3 Classic-path GPU cph (1d).** Force buffer-ops OFF; per-step full x+q H2D, potential D2H, host
  λ integration (unchanged). Flip the `nbnxm_setup.cpp:321` hard-fatal to allow `-nb gpu`.
  **Gate M1-GPU (GO/NO-GO):** single-point dV/dλ, GPU vs the PME oracle, ≤ ~1e-3 (single precision +
  atomics; confirm it is FP noise, not a systematic self-term error). **If this fails, the device force
  path is wrong — stop; residency is pointless.**

Layer 1 delivers a working (if transfer-bound) GPU cph and proves the device potential. It is exactly
the fork's capability, ported to 2026 GPU.

### Layer 2 — device-side O(Natoms) cph kernels (the residency enablers)
- **L2.1 Device λ-group tables (1–2d).** SoA device mirrors of, per group: `localAtomIndices` (CSR
  offsets + flat atom list), `localChargeDifferences`, `chargeA/chargeB`, `groupChargeIndices`,
  `x`(λ). Uploaded at setup and re-uploaded after each DD partition (NS steps only). Source of truth
  stays host `LambdaCoordinate` (`lambda_dynamics_params.h:88`).
- **L2.2 Device potential→groupPotential reduction (2d, per-backend).** One kernel: for each group,
  `Σ_a V[cells(a)]·Δq_a` (segmented reduction, one block/group or CUB segmented). Output the tiny
  `groupPotential[Ngroups]` device vector. Applies the nbat→atom mapping inline. **DD:** D2H the tiny
  `groupPotential`, `MPI_Allreduce` on host (or GPU-aware), H2D back — O(Ngroups) only.
- **L2.3 Device setLambdaCharges (2–3d, per-backend).** From updated λ (H2D'd tiny vector) compute
  per-atom charges and scatter into (a) the resident NB `xq.w` in nbat cell order and (b) the PME
  device charge buffer. Must reuse the existing charge-ordering maps (`atomIndices`/`cells`), never a
  raw `chargeA` memcpy. Handles constraint-group charge interpolation (mirror
  `constant_ph.cpp:1333`).

### Layer 3 — GPU-resident integration + guards + checkpoint
- **L3.1 Resident do_md wiring (2–3d).** On non-NS steps: after force reduction, run L2.2 (device) →
  groupPotential to host → host `updateLambdas` (the O(Ngroups) ODE, reused verbatim) → updated λ to
  device → L2.3 (device) refreshes charges **before** the next `do_force`. No x/v/f leave the device.
  The λ charge refresh must land before PME spread and NB — sequence it in the update stream.
- **L3.2 Workload decision + guards (1–2d).** In `taskassignment/decidegpuusage.cpp`
  (`decideWhetherToUseGpuForUpdate:690`, currently **no** cph awareness) allow `useGpuUpdate` with cph
  once L1–L2 land; keep it OFF while only Layer 1 exists (else host-λ + resident-update = per-step
  O(Natoms) transfer). Forbid the combos still unhandled (see §4): **separate PME ranks + cph**,
  **PME-on-different-GPU comm of the potential**.
- **L3.3 Checkpoint (0.5d).** λ x/v live in device tables during residency; D2H them before the
  existing (validated) checkpoint write. No format change.

### Layer 4 (optional) — full-device λ integrator
Port `computeForces` + v-rescale + constraints to device kernels, eliminating the O(Ngroups) host hop.
Only if profiling shows that hop matters (very large Ngroups). High risk (device RNG stream
reproducibility, coupled constraint solve) for small gain — **deliberately deferred.**

### Layer 5 — multi-backend replication
Only the kernels (L1.1, L1.2 gather, L2.2, L2.3) are per-backend; all host plumbing is backend-shared.
Replicate CUDA → SYCL/HIP/OpenCL after CUDA validates. Cluster is CUDA, so that is the priority.

---

## 4. Large-system generality — what must be explicitly handled or forbidden

- **PME is mandatory** for large systems → Layer 0/L1.2 are not optional here (unlike the RF project).
- **Separate PME ranks** (common at scale: PME on dedicated GPUs). The reciprocal `V_i` is produced on
  the PME rank and must travel back over PME→PP comm alongside forces. The fork **does not** send it
  (`ewald/pme_pp*` potential grep empty) — so separate-PME-rank cph is unsupported *even in the fork*.
  **Decision: forbid `npme>0` + cph initially** (clear fatal), add PME→PP potential plumbing as a
  dedicated later WP if a user needs it. Same-rank PME+PP+NB on one GPU is the supported large path.
- **DD across GPUs** is supported via L0.2 + L2.2's tiny cross-rank `groupPotential` reduce. The
  per-atom potential never crosses ranks — only the O(Ngroups) vector does.
- **GPU direct halo / PME decomposition** need no cph-specific work: `V_i` is a local per-home-atom
  quantity reduced per group; non-local atoms contribute via the existing force/halo machinery, and
  the group reduce sums over home atoms then Allreduces.

---

## 5. Risks & validation ladder

**Risks specific to variant B**
1. **Single precision + atomic summation** on `V_i` vs the double-ish CPU oracle → M1-GPU needs a
   looser tol; must confirm error is FP noise, not a wrong/misplaced self-term (the RF/Ewald self-term
   is the classic sign-error trap — keep it host-side in L1, as the CPU port does).
2. **Ordering bugs** — three orders in play (atom, nbnxm cell, PME grid). L2.2/L2.3 must route through
   `gridSet.cells()`/`atomIndices()`, never raw index. Highest-probability silent-wrong bug.
3. **Charge refresh ordering** — L2.3 must complete before PME spread + NB of the *next* step; a
   missed dependency gives stale-charge drift (the same class of bug WP-G0 fixed on CPU at nstlist>1).
4. **Determinism** — the hybrid keeps the RNG on host, so multi-step runs stay reproducible vs the CPU
   reference (a reason to prefer hybrid). Full-device (L4) would forfeit this.
5. **No local GPU** — slow iterate, full CUDA recompiles on aurum2; budget for it.

**Validation ladder** (oracle = CPU port = fork; all GPU on aurum2)
- **M0a/M0b** (Layer 0, CPU): PME dV/dλ vs fork ≤1e-4; DD run == single-rank.
- **M1-GPU** (Layer 1 gate): single-point PME dV/dλ, GPU vs CPU-PME oracle ≤~1e-3; every λ atom's
  `V_i` non-zero (catches the silent-zero failure).
- **M2-GPU** (Layer 3): multi-step at **nstlist=20** (production), resident GPU vs CPU — tests the
  device charge-refresh (L2.3/L3.1); trajectories track within FP/statistical bounds.
- **M3-GPU**: a full titration window — λ histograms / pKa agree with CPU within statistical error.
- **M-DD**: 2–8 GPU DD PME run reproduces single-GPU λ statistics; strong/weak scaling sanity.
- **Regression**: non-cph GPU runs bit-unchanged (all new buffers/kernels gated on `lambda_dynamics`).

---

## 6. Recommended sequence & honest scope

```
L0.1 PME potential (CPU)          ─┐ general-purpose correctness,
L0.2 multi-rank DD (CPU)          ─┘ independent of GPU — do first
      └─ M0a / M0b
L1.1 NB potential (GPU kernel)    ─┐
L1.2 PME potential (GPU)          ─┼ correct device potential
L1.3 classic-path GPU cph          ┘
      └─ M1-GPU  ← GO/NO-GO on the whole GPU effort
L2.1 device λ tables              ─┐
L2.2 device group reduce          ─┼ residency enablers
L2.3 device setLambdaCharges       ┘
L3.1 resident do_md wiring        ─┐
L3.2 workload guards              ─┼ ship GPU-resident cph
L3.3 checkpoint D2H                ┘
      └─ M2-GPU / M3-GPU / M-DD
L5   SYCL/HIP/OpenCL              (replicate kernels)
L4   full-device λ integrator     (optional, only if profiled necessary)
```

Rough total to shippable CUDA GPU-resident cph: **~4–6 weeks** of focused work, front-loaded by the
Layer 0 CPU generalization (PME + DD) that is valuable on its own. The M1-GPU gate after Layer 1 is the
real decision point: it proves the device force/potential path before any residency engineering is
committed.

**Scope honesty.** Layers 0–1 are a *port* (the fork already did this on CPU and GPU-PME); the novel
engineering is Layers 2–3 (the two device O(Natoms) kernels + resident wiring), and it is deliberately
kept small by the hybrid decision (§1). Full-device λ dynamics (L4) and separate-PME-rank cph (§4) are
explicitly out of the initial scope and gated behind clear fatals until a use case demands them.
