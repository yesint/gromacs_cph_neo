# GPU port plan: constant-pH (λ-dynamics) NB kernels → GROMACS 2026.1

Status: **PLAN / analysis only — nothing implemented yet.** This extends the completed CPU-NB
port (see `PORT_LOG.md`; CPU reference + SIMD kernels validated, M1 5e-5 / M2 vs fork). Goal:
make constant-pH run its non-bonded work on the **GPU** and still produce the correct per-atom
electrostatic potential that drives dV/dλ.

Target backend: **CUDA** first (the cluster is `spack load gromacs+mpi+cuda`; no local GPU, so all
GPU build/test happens on aurum2). SYCL/HIP/OpenCL are the *same design* — see "Multi-backend".

---

## 0. TL;DR architecture decision

Run cph on the GPU **with GPU buffer-ops DISABLED** (classic GPU-NB path: CPU force reduction,
full x+q host→device copy every step). This one choice makes three otherwise-hard problems free:

- **Per-step charge upload** — when GPU X-buffer-ops are off, `gpu_copy_xq_to_gpu` copies the full
  interleaved `xq` (coords **and** charges) H2D every step; the fast x→xq kernel that would otherwise
  *preserve* the stale `q` is not used. So refreshing λ-charges each step = the existing NS-step path,
  run every step.
- **Per-step potential copy-back** — add one ungated D2H of the new per-atom `potential` buffer in
  `gpu_launch_cpyback` (Local); on the `!useGpuFBufferOps` path this rides the local-stream
  `synchronize()` that already happens every step → **zero extra sync**.
- **do_md flow unchanged** — the CPU cph integration (setLambdaCharges → do_force → reduce → updateLambdas)
  stays exactly as today; only the NB kernel + the two transfers become GPU-aware.

The only thing that *must* live in device code is the pairwise potential accumulation inside the NB
kernel (it needs the neighbor loop). Everything else is host plumbing in **backend-shared** files.

---

## 1. How GPU NB offload works here (analyzed; file:line anchors)

**Device atom data** `NBAtomDataGpu` (`nbnxm/gpu_types_common.h:165`): `xq` = `DeviceBuffer<Float4>`
(x,y,z,**q** interleaved, `.w`=charge, :175); `f` = per-atom `DeviceBuffer<Float3>` (:179);
`eLJ`/`eElec` = size-**1** scalar energies (:182,184); `fShift` (:201). Passed to the kernel
**by value** (`cuda/nbnxm_cuda.cu:610` `prepareGpuKernelArguments`), so a new member reaches the
kernel with **no launch-signature change**.

**Kernel** `cuda/nbnxm_cuda_kernel.cuh` (760 lines). One block = one super-cluster `sci`
(8 i-clusters × 8 atoms), 64 threads = 2 warps, `tidxi`=i-atom, `tidxj`=j-atom. Compiled as **two
variants selected per step by `stepWork.computeEnergy`**: `_F_` (force only) and `_VF_` (force+energy)
(`nbnxm_cuda.cu:608`). RF pair energy (`K:655`, only under `CALC_ENERGIES`):
`E_el += qi*qj_f*(int_bit*inv_r + 0.5*two_k_rf*r2 - reactionFieldShift)` — i.e.
`coulFunc = int_bit/r + k_rf·r² − c_rf`. In-kernel **`qi = epsfac·q_i`** (shared `xqib`, scaled at
`K:342`), **`qj_f = q_j`** (raw global `xq`, `K:468`). Forces reduced via warp-shuffle +
`atomicAdd` into `atdat.f` (i: `reduce_force_i_warp_shfl` `utils:515`; j: `reduce_force_j_warp_shfl`
`utils:388`). Scalar energies reduced via `reduce_energy_warp_shfl` `utils:590` → `atomicAdd` into
size-1 `eLJ`/`eElec`.

**Per-step sequence in `mdlib/sim_util.cpp`**: x→xq / H2D (`:1775` convert, `:1799`
`gpu_copy_xq_to_gpu` when `!useGpuXBufferOps`) → `gpu_launch_kernel` (`kerneldispatch.cpp:490` →
`nbnxm_cuda.cu:516`) → `gpu_launch_cpyback` Local (`:1946` → `nbnxm_gpu_data_mgmt.cpp:1257`) →
`gpu_wait_finish_task` Local (`:2509` → local-stream `synchronize()` `gpu_common.h:334`) →
force reduce (`atomdata_add_nbat_f_to_f` `:2621`, CPU path) → `gpu_clear_outputs`
(`:2632` → `nbnxm_gpu_data_mgmt.cpp:1186`).

**Buffer lifecycle (backend-shared `nbnxm_gpu_data_mgmt.cpp`)**: `f` allocated in `gpu_init_atomdata`
(`:1050`), cleared **every step** in `gpu_clear_outputs` (`:1191`, unconditional); energies cleared
only under `computeVirial` (`:1196`). Energy D2H gated on `computeEnergy` (`:1356`); force D2H every
step on the `!useGpuFBufferOps` path (`:1315`). Host staging `NBStagingData nb->nbst`
(`gpu_types_common.h:143`); host energy reduce `gpu_reduce_staged_outputs` (`gpu_common.h:141`).

**Charge path** (the crux): charge is only ever written to device `xq.w` by the full H2D copy
`gpu_copy_xq_to_gpu` (`nbnxm_gpu_data_mgmt.cpp:1524`), which runs only when `!useGpuXBufferOps`
(NS steps). The fast x→xq buffer-ops kernel **deliberately preserves `q`**
(`cuda/nbnxm_gpu_buffer_ops_internal.cu:92`). Host reorders `mdatoms.chargeA` → nbnxm cluster order
via `nbnxn_atomdata_set_charges` using `gridSet.atomIndices()` (`atomdata.cpp:981`); padding q=0.
There is a ready but **UNUSED** `nonbonded_verlet_t::setAtomCharges` (`nbnxm.cpp:214`).

---

## 2. Current state — what's missing / wrong for GPU cph

- **No guard.** Nothing forbids `-nb gpu` with `lambda_dynamics` (grep of `taskassignment/` is empty).
  Today it runs, silently: the GPU kernel never fills `potential`, so `groupPotential=0` and λ never
  feels electrostatics — **silently wrong, not an error.** (`nbnxm_setup.cpp:561` sets the cph flag but
  only CPU kernels honour it; compile switch `kernel_common.h:59` is CPU-only.)
- **`setAtomCharges` is unwired (also a CPU-path gap).** `setLambdaCharges` writes `mdatoms.chargeA`
  every step (`md.cpp:1253`), but charges reach nbat only on NS steps via `setAtomProperties`
  (`sim_util.cpp:1437`, gated by `doNeighborSearch`). Production cph mdps use **`nstlist=20`**, so the
  CPU kernel and the dV/dλ currently use charges up to ~20 steps stale. WP5a intended to wire a
  per-step re-push (`setAtomCharges`) but it never landed. **Fix benefits CPU *and* GPU.**
- Legacy simulator is already forced for cph (`modularsimulator.cpp:425`); GPU update is not blocked
  by cph. So no simulator/guard removal is needed for GPU-NB.

---

## 3. Work packages

Effort assumes the CPU port author, now on the GPU side. Host plumbing (G0–G2, G4–G6) is
**backend-shared**, written once; only G3 (the kernel) is per-backend.

- **WP-G0 ✅ DONE (2026-07-07) — safety guard + per-step charge re-push.** (a) `pick_nbnxn_kernel`
  (`nbnxm_setup.cpp`) now **hard-fatals** if `lambda_dynamics` and the NB resource is Gpu/EmulateGpu
  (kills today's silent dV/dλ=0 path; message tells the user to run `-nb cpu`). (b) `do_force`
  (`sim_util.cpp`, after the non-NS `convertCoordinates`) now calls `nbv->setAtomCharges(mdatoms->chargeA)`
  every non-search step when `fr->constantPH` — the previously-unused `nbnxm.cpp:214` hook — so the NB
  kernel and dV/dλ always use current λ-charges. **Validated:** M1 unchanged (SIMD 1e-4, ref 1.4e-4);
  counterfactual at `nstlist=5` (NS at steps 0/5/10/15/20): with the fix the per-step dV/dλ trajectory
  is **bit-identical** to `nstlist=1`, whereas disabling the re-push makes it drift in a sawtooth
  (0 at NS steps, growing to ~3e-3 between them; ~2.3e-2 max over 20 steps) — i.e. the old code fed the
  kernel charges up to nstlist steps stale. This is the host hook the GPU path (G4) reuses.

- **WP-G1 (1d) — device `potential` buffer + lifecycle (backend-shared).** Add
  `DeviceBuffer<float> potential;` to `NBAtomDataGpu` (`gpu_types_common.h`, next to `f`), size
  `numAtomsAlloc`. Allocate/free next to `f` in `gpu_init_atomdata` (`:1050`/`:1025`);
  **clear every step** next to the unconditional `f` clear in `gpu_clear_outputs` (`:1191`) — *not*
  in the virial-gated block. Add pinned `HostVector<float> pot;` to `NBStagingData`
  (`gpu_types_common.h:143`), sized `numAtoms`, allocated near `:637`.

- **WP-G2 (1d) — per-step copy-back + host reduce (backend-shared).** In `gpu_launch_cpyback`
  (`nbnxm_gpu_data_mgmt.cpp`), inside the Local section but **outside** the `computeEnergy` guard, add
  an async D2H `copyFromDeviceBuffer(nb->nbst.pot.data(), &adat->potential, 0, numAtoms, stream, …)`.
  Gate the whole thing on the cph flag so non-cph GPU runs pay nothing. Add a GPU reduce path: a small
  `reduceElectrostaticPotentialGpu` (or extend `gpu_reduce_staged_outputs`, `gpu_common.h:141`) that
  maps `nbst.pot` (nbat order) → atom order via `gridSet.cells()` (same mapping the CPU
  `reduceElectrostaticPotential` uses, `atomdata.cpp:1628`) into `fr->electrostaticPotential`, and
  **adds the RF/Ewald self-term on the host** (`potential[a] -= facel·q_a·c_rf`, per real atom). Doing
  the self-term host-side keeps the kernel edit minimal and reproduces the CPU kernel's self-term.

- **WP-G3 (2–4d, HIGH — the crux, per-backend) — kernel potential accumulation.** In
  `cuda/nbnxm_cuda_kernel.cuh`: expose `float* potential = atdat.potential;` and
  `float c_rf = nbparam.c_rf;` **outside** the `CALC_ENERGIES` guards (so both `_F_` and `_VF_`
  variants have them). Immediately after the force pair update (`K:665–671`), unconditionally, when
  the pair is in range/interacting:
  ```
  float coul = int_bit*inv_r + 0.5f*two_k_rf*r2 - c_rf;   // = 1/r + k_rf r^2 - c_rf, exclusion-masked
  atomicAdd(&potential[ai], nbparam.epsfac * qj_f * coul); // V_i += facel*q_j*coulFunc
  atomicAdd(&potential[aj], qi            * coul);         // V_j += facel*q_i*coulFunc
  ```
  (`ai = ci*c_clusterSize + tidxi`, `aj` in scope at `K:462`; each unordered pair is visited exactly
  once — the force Newton-pair update proves it, so writing both endpoints is correct, no double count;
  `int_bit`/`nonSelfInteraction` reproduce exclusion + self masking). Self-term done host-side (WP-G2),
  so no preload edit needed. Gate the accumulation on a kernel `bool computePotential` arg (uniform
  branch, cheap) **or** a new `_F_pot_`/`_VF_pot_` kernel-table dimension — prefer the runtime bool to
  avoid a kernel-variant explosion, mirroring the CPU runtime `computePotential`.
  **Perf-optimised variant (do after correctness):** replace the two per-pair `atomicAdd`s with
  register accumulators — `poti_buf[c_superClusterSize]` (like `fci_buf`) + scalar `potj_buf` (like
  `fcj_buf`) — reduced by scalar analogues of `reduce_force_i/j_warp_shfl` and one `atomicAdd` each.

- **WP-G4 (0.5d) — wire the GPU path into do_md.** On GPU-NB cph steps, ensure the charge re-push
  (G0b) targets the GPU (`setAtomCharges` updates host nbat `x()`; forcing `!useGpuXBufferOps` makes
  `gpu_copy_xq_to_gpu` push it) and the GPU reduce (G2) fills `fr->electrostaticPotential` before
  `updateLambdas` (`md.cpp:1303`). Keep B1/B5/B6 (setLambdaCharges, updateLambdas, updateAfterPartition)
  host-side unchanged.

- **WP-G5 (0.5d) — force the classic GPU path for cph.** When `lambda_dynamics` + GPU NB: set
  `useGpuXBufferOps=false` and `useGpuFBufferOps=false` in the workload decision
  (`decidesimulationworkload.cpp`), and forbid separate PME ranks / GPU-resident update for cph
  (single-rank scope). This is what makes G1–G4 "free" (per-step full xq H2D, per-step force+pot D2H,
  the every-step local sync). Flip the G0 hard-fatal into "GPU cph enabled".

- **WP-G6 (per-backend replication) — SYCL / HIP / OpenCL.** Only the G3 kernel snippet repeats, in
  `sycl/nbnxm_sycl_kernel_body.h`, `hip/…`, `opencl/nbnxm_ocl_kernel.clh` (same math, same buffer
  which is already backend-shared). Do only if those backends are actually used (cluster = CUDA).

---

## 4. Validation (oracle = the CPU port / fork; GPU built on aurum2)

- **M1-GPU (go/no-go):** the locked M1 single-point tpr, `mdrun -nsteps 0 -nb gpu` on aurum2 →
  `GMX_CPH_DUMP_DVDL` per-group dV/dλ vs `M1_oracle_dvdl.txt`. GPU is single-precision with atomic
  summation → expect a **looser tol than CPU** (target ≤ ~1e-3 abs on ~200-magnitude values, i.e.
  relative ~1e-5; compare also directly against the CPU-SIMD dvdl on the same inputs). Confirm per-atom
  potential is non-zero for all λ atoms (catches the silent-zero failure).
- **M2-GPU:** short multi-step at **nstlist=20** (the production value) vs the CPU run — this is the
  real test of the per-step charge re-push (G0b/G4): without it the two diverge between NS steps.
- **M3-GPU:** one full US/cph window CPU vs GPU — λ histograms / titration agree within statistical error.
- **Regression:** non-cph GPU runs unchanged (guard the new buffer/copy/kernel work on the cph flag).

---

## 5. Risks & the honest cost/benefit

**Risks.** (1) GPU atomic contention on `potential` in the hot loop (mitigate: warp-shuffle reduction,
WP-G3 optimised variant). (2) Kernel-variant explosion if done at compile time (mitigate: runtime
`computePotential` bool). (3) Atom-ordering: must route λ-charges through `setCharges`/`atomIndices()`,
never memcpy `chargeA` to device (Agent-verified). (4) Single-precision GPU vs the CPU oracle — needs a
looser M1 tol; verify it's FP noise, not a systematic term (esp. the self-term sign/placement).
(5) No local GPU → slow iterate + full CUDA recompiles on aurum2. (6) PME-on-GPU would miss the
reciprocal potential — **fine for Martini RF (no PME)**; forbid GPU-PME-cph explicitly.

**Is it worth it for THIS project?** Be clear-eyed:
- The system is small (46k beads), Martini **RF (no PME)**, so NB is essentially the whole force calc.
  CPU-SIMD cph is already ~4 ms/step NB (WP5b). A GPU NB kernel would be fast, but the classic path
  adds a full x+q H2D + f D2H + pot D2H + a per-step sync each step (~hundreds of µs of PCIe + launch),
  which for a system this small is a large fraction of the step.
- The production campaign runs **many walkers sharing each GPU** (job-middle: 32 walkers / 2 GPUs =
  16/GPU). Under that sharing, per-walker GPU throughput is far below dedicated, and CPU-SIMD (each
  walker on its own cores) likely gives **better aggregate throughput**.
- **GPU cph pays off when:** systems get larger, few/one walker per GPU (dedicated), the CPU is the
  bottleneck or needed elsewhere, or you want maximum single-window wall-clock.

**Recommendation.** Do **WP-G0 now regardless** (the guard + per-step charge re-push — it closes a real
CPU correctness gap and is cheap). Treat WP-G1…G6 as a strategic capability: implement when a
larger-system or dedicated-GPU use case appears; for the current shared-GPU 480-window campaign, CPU
SIMD is likely the better production choice. Sequence to a go/no-go: **WP-G0 → G1 → G2 → G3 → M1-GPU**
(if dV/dλ matches, commit to G4/G5/M2–M3; else the force path is wrong and stop).

---

## 6. Multi-backend note
Host plumbing (buffer, clear, copy-back, charge push, reduce, guards, do_md wiring — WP-G0/1/2/4/5)
lives in **backend-shared** files (`gpu_types_common.h`, `nbnxm_gpu_data_mgmt.cpp`, `nbnxm.cpp`,
`sim_util.cpp`, `md.cpp`) → written once, works for all backends. Only the WP-G3 kernel snippet is
duplicated per backend. Target/validate **CUDA** (the cluster); replicate to SYCL/HIP/OpenCL only if needed.
