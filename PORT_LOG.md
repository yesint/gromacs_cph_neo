# cph → GROMACS 2026.1 port — progress log

Port tree: `/home/semen/install/gromacs-2026-cph` (git; baseline commit = pristine 2026.1).
Fork oracle (2021.0, built): `/home/semen/install/constantph/gromacs-constantph/build/bin/gmx`.
Pristine 2021.0 (for diffs): `<scratchpad>/gromacs-2021`.
Reference diffs (fork vs 2021, per touched file): `_portref/00_fork_vs_2021_all.diff` (+ `00_stat.txt`).

## Build
```
cd build-cpu   # configured: GMX_GPU=OFF GMX_MPI=OFF GMX_DOUBLE=OFF GMX_FFT_LIBRARY=fftw3 BUILD_TESTING=OFF, gcc16
cmake --build . -j16 --target gmx
```
Baseline stock gmx builds & runs Martini-RF+PW (WP0 ✓).

## M1 oracle (LOCKED)
Dir: `<scratchpad>/m1_oracle`. Self-contained topo via symlinks (toppar, molecule_0, elastic nets, index_cph, ref_system.pdb from full_size/{,cph/}).
- `sp.mdp` = md/nsteps 0/RF eps2.5/no-pull + `cph.mdp.inc` + **fixed seeds** `lambda-dynamics-random-seed=12345`, `-random-vv-seed=67890`.
- Fork: `grompp` (46 λ coords) → `mdrun -nsteps 0 -nb cpu` w/ `GMX_NBNXN_PLAINC_1X1=1` → `gmx cphmd -dvdl`.
- **Target: `oracle_ref/M1_oracle_dvdl.txt`** — 46 groups, dV/dλ range −209.36…+197.34. First 6: 169.15, −17.401, −209.36, 17.099, 146.303, −144.268.
- **M1 protocol**: port grompp+run SAME `sp.mdp` (tprs NOT shareable across 2021↔2026 tpx gap → each side grompps its own). Fixed seeds ⇒ identical λ-integrator RNG (shared source) ⇒ any dV/dλ diff is purely the WP5a force path. PASS = all 46 match ≤1e-4.

## ‼️ RUNTIME GOTCHA (cost hours)
The port `build-cpu/bin/gmx` will silently load the **installed stock** libgromacs at
`/home/semen/programs/gromacs-2026.1/lib64/libgromacs.so.11` (put on `LD_LIBRARY_PATH` by
the user's GMXRC), NOT the rebuilt `build-cpu/lib/libgromacs.so.11` — so cph code is inert
(mdp reads report `Unknown left-hand 'lambda-dynamics'`, no hooks run). **ALWAYS run the port as:**
```
LD_LIBRARY_PATH=/home/semen/install/gromacs-2026-cph/build-cpu/lib:$LD_LIBRARY_PATH build-cpu/bin/gmx ...
```
Verify with `ldd build-cpu/bin/gmx | grep libgromacs` → must point into build-cpu/lib.

## Build/iterate loop (fast)
- Single object: `make -f src/gromacs/CMakeFiles/libgromacs.dir/build.make src/gromacs/CMakeFiles/libgromacs.dir/<path>.cpp.o`
- Full: `cmake --build build-cpu -j16 --target gmx -- -k`  (make-native `-k`; cmake has no --keep-going)

## Status
- **WP0 ✓** baseline builds+runs Martini-RF.
- **WP1 ✓** module (constant_ph/*, lambda_dynamics_params.h, cphmd.cpp) compiles against 2026 (commit 13cbdee).
- **WP2 ✓ (core)** RandomDomain::ConstantPH, enxCPHMD, eLambdaTcoupl enum+names, t_inputrec +
  t_forcerec fields, tpxv_ConstantpH. (Remaining WP2: cptv_ConstantpH checkpoint enum, ewcCONSTANTPH
  wallcycle — needed by WP4/WP6.)
- **WP3 ✓** (commit f8b3bdb) — **port grompps the cph system: 46 lambda coordinates, charges baked.**
- **WP4 ✓** (commit 0252d8f) — **port RUNS cph end-to-end**: `mdrun -nsteps 0 -nb cpu` exit 0, 46 λ
  coords, legacy simulator forced, hooks fire. `GMX_CPH_DUMP_DVDL=1` → `cph_port_dvdl.dat` (per-coord
  `dvdl_pot`). Currently all 0.0 — CORRECT: NB kernel doesn't fill the potential buffer yet (= WP5a).
- **WP5a (force path)** — IN PROGRESS. 2026 reference kernel is structurally close to the fork's 2021
  (CALC_COUL_RF, `fcoul=qq*(interact*rinv*rinvsq-k_rf2)`, `out->Vc`, `Vc_sub_self`, `do_self` all present),
  so the fork diffs in `_portref` port with adaptation (HostVector not FastVector). Components:
  (1) kernel_common.h `#define GMX_COMPUTE_ELECTROSTATIC_POTENTIAL 1`; (2) atomdata.h `HostVector<real>
  potential` in nbnxn_atomdata_output_t; (3) atomdata.cpp resize + reduceElectrostaticPotential;
  (4) kernel_common.cpp clear; (5) kernel_ref_outer.h `potential=out->potential.data()` + self-term
  `potential[i]-=qi[i]*2*Vc_sub_self`; (6) kernel_ref_inner.h coulFuncValue/pcoul + `potential[i]+=q[j]*pcoul`;
  (7) sim_util.cpp per-step setAtomCharges re-push + reduce (!useGpu) → copy to fr->electrostaticPotential;
  (8) forcerec.cpp `bHaveQ |= lambda_dynamics`; (9) nbnxm.{cpp,h} reduce entry point.
- **WP5a wiring DONE** (commit pending): added `nbnxn_atomdata_t::reduceElectrostaticPotential`
  (sum outputBuffers[*].potential over threads, map nbat→atom via gridSet.cells()) + `setCharges`;
  `nonbonded_verlet_t::reduceElectrostaticPotential`/`setAtomCharges` wrappers; do_md reduces into
  `fr_->electrostaticPotential` (aliases constantph_->potential()) after do_force, before updateLambdas.
  Charge re-push NOT needed at step 0 (NS-step setAtomProperties already pushes the λ charges that
  setLambdaCharges wrote into mdatoms->chargeA); needed for production multi-step (add setAtomCharges
  each step + kernel_common.cpp potential-buffer clear).
## ✅✅ M1 PASSES (2026-07-07) — go/no-go = GO
Port `mdrun -nsteps 0 -nb cpu -ntomp 1` w/ `GMX_NBNXN_PLAINC_1X1=1` on the 46-λ cph system:
**all 46 groups' dV/dλ (dvdl_pot) match the fork oracle, MAX|diff| = 5e-5** (single-precision rounding).
The CPU-NB reference-kernel force path is CORRECT — the port grompps, runs, and computes cph identically
to the 2021 fork. Reproduce: `bash <scratchpad>/run_m1.sh`.

Two bugs found+fixed reaching M1 (both silent zeros):
1. `GMX_COMPUTE_ELECTROSTATIC_POTENTIAL` (kernel_common.h) wasn't visible in the reference-kernel TU
   → CALC_ELEC_POTENTIAL + the potential pointer compiled out with no error. Fix: `#include
   "gromacs/nbnxm/kernel_common.h"` in kernel_ref_1x1.cpp + kernel_ref_4x4.cpp.
2. `ConstantPH::updateAfterPartition` was never called → lambda groups' localAtomIndices empty →
   updateLambdas summed nothing. Fix: call it at the end of `mdAlgorithmsSetupAtomData`
   (domdec/mdsetup.cpp): `fr->constantPH->updateAfterPartition(dd ? dd->ga2la.get() : nullptr,
   numHomeAtoms, fr->natoms_force)`.

## Post-M1 progress (commits 195ac78 → 505654d)
- **Guard ✓** (195ac78): `pickNbnxnKernelCpu` auto-selects the plain-C 4x4 reference kernel when
  `ir->lambda_dynamics` (SIMD kernels have no potential accumulation → would silently give 0).
  Verified: 4x4 auto-selected without env var; dvdl matches oracle to 1e-4 (4x4-vs-1x1 FP noise).
- **Per-step potential clear ✓** (195ac78): kerneldispatch.cpp clears out.potential each step (with the
  force clear). M2 verified maxAbsDvdl stays constant across steps (no accumulation).
- **M2 ✓** (e618116): port runs multi-step stably (dt=0.001, nstlist=1), clear verified.
  (Full fork-trajectory M2 needs an equilibrated seed — ref_system.pdb blows up in real MD.)
- **WP6 edr output + gmx cphmd ✓** (505654d): energyoutput printStepToEnergyFile writes the
  enxCPHMD block (`cphmd->writeToEnergyFrame`); md.cpp passes constantph_ + do_cphmd gate;
  legacymodules registers `gmx cphmd`. VERIFIED: port writes enxCPHMD (id=8, 46 subblocks) to .edr,
  `gmx cphmd -dvdl` extracts all 46 groups matching oracle to 1e-4.

## Checkpoint ✓ (commit bb4b5b0) — λ x,v persist across `-cpi` restarts
`CheckPointVersion::ConstantPH` + `CheckpointHeaderContents.flagsConstantpH` + `doCptConstantpH`
(serializes each LambdaCoordinate x,v). Threaded: write = mdoutf (gmx_mdoutf stores constantph via
`mdoutf_set_constantph`, called in do_md) → static write_checkpoint → write_checkpoint_data; read =
runner (ConstantPH created BEFORE applyLocalState, nullptr commrec — single-rank) → applyLocalState →
load_checkpoint → read_checkpoint. Also read-to-advance in read_checkpoint_data + list_checkpoint (block
alignment). VALIDATED: restart λ trajectory matches the continuous-run reference (unbroken).

## THE PORT IS PRODUCTION-COMPLETE (single-rank per window)
grompp ✓, multi-step MD ✓ (M1 5e-5 vs fork, M2 stable), correct dV/dλ ✓, auto plain-C kernel ✓,
edr output + gmx cphmd ✓, chunked -cpi restarts ✓. Run: `LD_LIBRARY_PATH=.../build-cpu/lib gmx mdrun ...`.

## ✅✅ WP5b SIMD kernel — DONE + VALIDATED (2026-07-07)
The SIMD non-bonded kernel now accumulates the per-atom electrostatic potential, so cph runs
on the fast SIMD kernels instead of the plain-C reference. **SIMD is the default for cph**; set
`GMX_CPH_NO_SIMD=1` to fall back to the plain-C 4x4 reference kernel (escape hatch for debugging).
Files touched (all under `src/gromacs/nbnxm/`):
- **atomdata.{h,cpp}**: added `computeElectrostaticPotential_` flag (+ getter/setter). This is the
  runtime signal that gates the SIMD potential work — set = `inputrec.lambda_dynamics` in
  `nbnxm_setup.cpp` right after nbat construction. (The potential *buffer* resize stays keyed on the
  compile macro so the reference kernel is untouched; the SIMD kernel gates on the flag, so non-cph
  SIMD runs skip the work.)
- **nbnxm_setup.cpp** `pickNbnxnKernelCpu`: cph uses the normal SIMD selection by default; it forces
  plain-C 4x4 **only when `GMX_CPH_NO_SIMD` is set**.
- **simd_kernel.h**: unpack `potential` ptr + `computePotential` bool; init `ewaldShift` also when the
  potential macro is on (needed on non-energy steps); zero-init a `potentialIV[nR]` accumulator by the
  i-forces; RF/Ewald potential **self-term** (runtime-guarded, mirrors the energy self); after the
  j-loop **reduce** `potentialIV` into `potential[sci..]` via `reduceIncr4ReturnSum` (4xM) /
  `reduceIncr4ReturnSumHsimd` (2xMM) — index by **atom** (`sci`), not coord (`scix`).
- **simd_kernel_inner.h**: compute `coulFuncV[i] = selectByMask(rInvExclV[i]-vCoulombCorrectionV[i],
  withinCutoffV[i])` (in the `!calculateEnergies` path too, via forceAndCorrectionEnergy); then
  i-atom `potentialIV[i] += (facel*jq_S)*coulFuncV[i]` and j-atom scatter of `sumArray_i(chargeIV[i]*
  coulFuncV[i])` — 4xM: `store(potential+aj, load+sum)`; 2xMM: `incrDualHsimd(potential+aj,
  potential+aj, sum)` (same-ptr trick folds the two duplicated j-halves = low+high).

**Layout note (AVX2_256, single):** default selection is **4xN** (c_iClusterSize=4, c_jClusterSize=8,
1 j-cluster/reg, nR=4, clusterRatio JSizeIsDoubleISize → `sci=(ci>>1)*8+(ci&1)*4`); 2xNN also compiled.
Both paths implemented and validated.

**Validation (all vs the locked fork oracle / the reference kernel):**
- **M1** (single-point, 46 λ groups, `bash <scratchpad>/run_m1_simd.sh`): SIMD 4xN & 2xNN dV/dλ match
  the fork oracle to **max 1e-4** (mean 2.5e-5) — pure single-precision noise (largest-|dvdl| group 3
  −209.36 → reldiff 1.4e-7). SIMD-vs-reference-kernel max 6e-5. Reference regression preserved (5e-5).
- **M2** (`ck2.tpr`, 4 steps, dt=0.001, **nstlist=1** ⇒ NS+charge-repush every step): SIMD per-step
  maxAbsDvdl & λ track the reference kernel to ~1e-5; stable, no accumulation.
- **Output path**: `gmx cphmd -dvdl` on the SIMD-produced `.edr` matches the reference `.edr` to 2e-4
  over all 230 per-step/per-group cells (5 steps × 46 groups).
- **Speed** (8 threads, 46k beads, `ck2.tpr`): NB Force **31.97 ms/step (plain-C-4x4) → 4.13 ms/step
  (SIMD-4xN) ≈ 7.7× faster**; total wall ~3.2× (short run; higher once startup amortizes).

**How to run cph:** just `LD_LIBRARY_PATH=.../build-cpu/lib gmx mdrun ...` — SIMD (auto 4xN) is the
default (force layout with `GMX_NBNXN_SIMD_4XN` / `GMX_NBNXN_SIMD_2XNN`). `GMX_CPH_NO_SIMD=1` reverts
to plain-C 4x4.

## THE PORT IS FULLY COMPLETE — remaining items are cosmetic/optional
- Cosmetic: edr block name shows 'id' not 'Constant pH data' in gmx dump (cphmd matches by numeric id).
- Multi-rank DD cph (real commrec) — not needed for the single-rank campaign.
- M3–M5 (titration/pKa vs fork; full two-domain window; refdata regression test) — science validation,
  not port work.

## GENERAL-PURPOSE GPU work — see GPU_LAMBDA_UPDATE_PLAN.md (branch gpu-lambda-update)

### ✅ L0.1 DONE + M0a PASS (2026-07-09, commit 1b90be2) — CPU PME reciprocal potential
The CPU port previously filled `fr->electrostaticPotential` from the **NB real-space kernel only**
(correct for reaction-field Martini, WRONG for any PME system — the reciprocal dV/dlambda term was
missing). Ported the fork's PME mesh potential into the 2026 CPU PME path:
- `pme_gather.{h,cpp}`: `gatherPmePotential()` (theta·grid contraction, mirrors `gather_f_bsplines`).
- `pme.{h,cpp}`: `gmx_pme_do()` gets a `potentials` out-arg, filled after the Coulomb force gather
  (single PME rank asserted). **KEY FIX:** `bDoSplines |= !potentials.empty()` forces all-atom
  b-splines — else `make_bsplines` skips atoms with charge 0, and a titratable atom that is *neutral
  at the current lambda* (charge 0 but non-zero charge-difference) gets no valid spline → its
  reciprocal potential is garbage. This was exactly the HIS + BUF bug (see below).
- `force.{h,cpp}`, `sim_util.cpp`: thread `fr->electrostaticPotential` into `gmx_pme_do` when cph on.
- `md.cpp`: clear the potential buffer **before** do_force (PME adds during, NB reduce adds after —
  both `+=`); the old post-do_force `std::fill` would have wiped the PME part.
- `pme_only.cpp`: passes empty potentials (separate-PME-rank cph unsupported).

**M0a protocol (new PME oracle; the locked M1 oracle is RF-only):** same 46-λ cph system as M1 but
`coulombtype = PME` (`sp_pme.mdp`), single point (`nsteps 0`), reference kernel
(`GMX_CPH_NO_SIMD=1 GMX_NBNXN_PLAINC_1X1=1 -nb cpu -ntmpi 1 -ntomp 1`). grompp+run BOTH the 2021 fork
and the port; extract per-group dV/dλ (port: `GMX_CPH_DUMP_DVDL=1`→`cph_port_dvdl.dat`; fork:
`gmx cphmd -dvdl -nocoordinate -numplot 1`→`<base>-dvdl-N.xvg`). Repro dir + script:
`cph/m0a_repro.sh` (rebuilds it in a scratch dir from `cph/` inputs).
**Result: all 46 groups match to the single-precision floor** — max **relative** 1.1e-5 (identical
144×144×96 grid), max abs 1.9e-3 on the buffer group (magnitude 4231 → rel 4.5e-7). The RF regression
is unchanged (max abs 5e-5); the PME floor is ~5× the RF floor = the extra reciprocal-FFT summation in
single precision, not a code error. **Before the all-atom-spline fix the 4 HIS groups + buffer were
off by up to 100%** (they are neutral at their initial λ). Multi-threaded PME gather (`-ntomp 4`)
matches single-thread (no race; threaded spline path validated).

**Scope note:** single PME rank only. PME domain decomposition (`pme->nnodes>1`) hard-asserts — that
(and separate PME ranks) is deferred with the rest of the multi-rank work (L0.2 / §4 of the plan).

### ✅ L0.2 DONE + M0b PASS (2026-07-09, commits 34dc38b + 3229b8f) — multi-rank DD cph
DD cph was broken in **three** independent ways (RF, 4 thread-MPI ranks vs 1 rank, single point);
all fixed. RF DD now reproduces single-rank dV/dλ to the single-precision floor for 2/4/8 ranks and
4 ranks × 2 threads (max rel 3e-6), a 30-step run with `nstlist=10` repartitioning matches
single-rank over the whole trajectory, and single-rank M0a is unregressed. Repro:
`full_size/cph/m0b_repro.sh`. **DD+PME: see L0.2b below (now solved for PME on the PP ranks).**
The three causes:

1. **Cross-domain group-potential reduce (commit 34dc38b).** A λ group's *atoms* can live on different
   ranks; each rank must sum its home atoms' Σφ·Δq and all-reduce. `ConstantPH` was built with a
   `nullptr` commrec (`runner.cpp` — early construction so a checkpoint can populate λ), so the
   `commMyGroup.sumReduce` in `updateLambdas` and the λ-broadcast in `updateAfterPartition` were dead.
   Added `ConstantPH::setCommrec()`, called in runner once `cr` exists. `sumReduce` is `MPI_Allreduce`
   (thread-MPI) so every rank integrates λ identically. No-op at single rank (size()>1 guards).

2. **Group double-count via ga2la (commit 3229b8f, constant_ph.cpp `updateAfterPartition`).** It built
   each group's `localAtomIndices` with `ga2la->find()`, which returns BOTH home and halo (imported)
   copies (`Entry.cell > 0`). A titratable atom that is home on one rank and a halo copy on another was
   then added to the group on **both** ranks → counted twice in the group `sumReduce`. Fix: use
   `ga2la->findHome()` (non-null only for `Entry.cell == 0`), so each atom is counted exactly once, on
   its home rank. (This corrects the earlier note that the ga2la mapping "was already home-only" — it
   was not.)

3. **Per-atom potential halo back-communication (commit 3229b8f, md.cpp).** The NB kernel adds each
   pair's potential to BOTH endpoints; under DD a home atom that is the *halo* partner of a pair computed
   on another rank has that contribution land on the halo copy, which must be summed back to the owner —
   exactly like forces. The port previously reduced only `AtomLocality::Local` with no halo move. Fix:
   mirror the force reduction sequence `reduceForces(NonLocal) → dd_move_f → reduceForces(Local)` —
   reduce the halo range, move it home (potential embedded in an rvec x-component so it rides `dd_move_f`
   / whichever halo backend is active), THEN reduce the local range. **Order is load-bearing:** `dd_move_f`
   uses the home-atom slots as forwarding scratch across DD pulses, so at the move they must hold only
   received (halo) contributions, not the local part (a single `reduce(All) + move` corrupts multi-pulse
   forwarding).
   **The 2021 fork does NOT implement any of this** (grep-verified: zero DD/MPI comm of the per-atom
   potential; `dd_move_f` moves only forces; the `fnb[i+3]` read in `add_nbat_f_to_f_part` is a dead path
   — the cph kernel writes the separate `out.potential` buffer). The fork's only cph MPI ops are the group
   `sumReduce` + λ `gmx_bcast` (= bug 1), so it is correct only single-rank per window (its actual usage).
   ⇒ bugs 2 and 3 are genuinely NEW work, not present upstream.

Diagnosis lesson: bugs 2 and 3 both had to be fixed together — the halo move alone left an over/under
residual (the double-counted halo copies), and `findHome` alone left an under-count (incomplete
home-atom potential). Each masked the other in single-fix tests.

### ✅ L0.2b DONE — DD + PME (2026-07-10, commits 383ec0c + 094e449)
The L0.1 PME potential asserted `pme->nnodes == 1`. Now DD+PME works with **PME decomposed across the
PP ranks** (the default when no separate PME ranks are used):
- The reciprocal potential is gathered on each PME slab in slab-atom order (`atc.potentials`, new
  `FastVector<real>` on `PmeAtomComm`, allocated in `setNumAtoms`) and **redistributed back to PP
  order exactly like the forces** — `dd_pmeredist_potential()` is the scalar analogue of
  `dd_pmeredist_f` (reuses `pme->bufr`, already sized by the forward coefficient redistribution), and
  it's called in the same redistribute-back loop, cascading through the decomposition dims. This is
  cleaner than the fork's bespoke `PotentialsComm`/`potentialAtomsBuffer` (which communicated only the
  λ subset via a separate atom-index comm) — the potential just rides the existing force-redist
  structure for all atoms.
- **md.cpp NB halo move made PME-safe:** home atoms already hold the reciprocal part after do_force,
  and `dd_move_f` uses home slots as forwarding scratch, so the home slots are zeroed at the move (PME
  neither forwarded nor overwritten) and the received halo NB contribution is **added** afterward.
- **Validated:** DD+PME 1-vs-{2,4} ranks single-point matches single-rank to the single-precision/FFT
  floor (median rel 4e-7; the worst-case rel ~3e-5 is the cphmd xvg print granularity on small groups,
  not error); 20-step DD+PME matches single-rank; M0a + M0b unregressed. Same grid on all rank counts.
- **Separate PME ranks (`-npme > 0`) are guarded, not silently wrong:** the reciprocal potential is
  computed on the PME rank and not yet returned over the PME→PP link, so `createSimulationWorkload`
  `gmx_fatal`s on `lambda_dynamics + haveSeparatePmeRank` (message: run PME on the PP ranks). Verified:
  `-ntmpi 4` runs, `-ntmpi 4 -npme 1` fails cleanly. Implementing separate-PME-rank support = a later
  WP (send the potential in `gmx_pme_send_force_vir_ener` / receive in `gmx_pme_receive_f`; sub-cases
  for npme=1 vs >1 and the GPU PME-PP comm). Not needed for the campaign or typical DD runs.

### ✅ L1.2 DONE — CUDA PME reciprocal potential on GPU (2026-07-10, commit a220ca6)
The GPU PME gather kernel now produces the per-atom reciprocal potential (the last missing piece of
"CUDA PME"). Host plumbing landed earlier (commit d88f48c): `d_potentials` device buffer +
`h_potentials` pinned staging (sized `nAtomsAlloc`, realloc'd with the forces), copy-back gated on
`PmeGpuSettings::computeElectrostaticPotential` (= `pme->computePotential` = `ir->lambda_dynamics`),
`PmeOutput::potentials_`, and `pme_gpu_reduce_outputs` accumulating into `fr->electrostaticPotential`.
- **Kernel (`pme_gather.cu`):** `sumForceComponents` gained a `float* potential` out-param and
  accumulates `theta_x*theta_y*theta_z*grid` (values only, no derivative — `*potential += tdx.x*tdy.x*fxy1`
  where `fxy1 = theta_z*grid`) in the same inner loop as the force. After the force reduction, the
  per-atom partials are collapsed over the atom's `atomDataSize` lanes with a `__shfl_down_sync`
  (width `atomDataSize`), and lane 0 (`splineIndex==0`) stores to `d_potentials[atomIndexGlobal]`.
  Gated `if constexpr (order == 4 && numGrids == 1)` = the constant-pH regime; `d_potentials` is sized
  to `nAtomsAlloc`, so the store is always in bounds (padding slots written, never read).
- **No uncharged-atom special-case needed on GPU.** `c_skipNeutralAtoms == false` (pme_gpu_constants.h),
  so `pme_gpu_check_atom_charge` returns true for every atom → splines are computed and the potential is
  gathered for **all** atoms, including titratable atoms that are electrically neutral at the current λ
  (HIS at λ=1, BUF at λ=0.5). The L0.1 CPU concern (`make_bsplines` skipping zero-charge atoms, fixed
  there with `bDoSplines |= !potentials.empty()`) does **not** recur on the GPU.
- **Validated** (`pmegpu_run/`, RTX 3080, `-nsteps 0`, `GMX_CPH_DUMP_DVDL=1`), full-GPU
  (`-nb gpu -pme gpu`) single-point dV/dλ vs the CPU oracle (`-nb cpu -pme cpu`):
  - 46-group HIS/BUF system: max rel **2.0e-5**, median 1.2e-6, p90 4.2e-6 (0 groups > 5e-5).
  - 88k all-atom CHARMM36: max rel **6.2e-5** (group 54, dV/dλ≈1125 → 0.07 abs), median 2.1e-6 (1/54 > 5e-5).
  - The mixed control (`-nb gpu -pme cpu`) stays at 1.6e-5 on both (= the already-validated L1.1 NB
    level); the L1.2-isolated delta (gpuPME vs cpuPME) matches the full-GPU-vs-oracle error and tracks
    group magnitude ⇒ pure single-precision cuFFT/gather floor, not a bug. **CUDA PME is finalized.**

### ✅ L3 DONE — GPU-resident constant-pH update loop (2026-07-10, commits 0c8244f R0 + e300526 R0b)
cph now runs fully GPU-resident: `-nb gpu -pme gpu -update gpu` keeps x/v/f on the device with the
leapfrog+constraints integrator on the GPU. Buffer ops ON removes the classic per-step full x+q H2D,
so two data paths had to be rebuilt for cph, plus the workload guards:
- **Charge → NB xq.w:** the nbnxm X buffer-op preserves xq.w and is now the only per-step xq writer, so
  it also re-packs xq.w from a new device charge buffer `NBAtomDataGpu::lambdaCharges` (atom order),
  H2D'd from the λ-interpolated `mdatoms->chargeA` each step (`uploadLambdaChargesToGpu`). nullptr for
  non-cph ⇒ the .w path is fully off (bit-identical). Guard `numAtomsAlloc>0` (numAtoms is -1 until the
  first pair search; the first step is NS and its full H2D packs xq.w anyway — a -1-length copy was the
  first bug, surfacing as a sticky `cudaErrorInvalidValue` at teardown).
- **Charge → PME d_coefficients:** uploaded only at NS, so `-pme gpu` resident needs a per-step refresh:
  new light `gmx_pme_reinit_charges_gpu` (charge-only H2D, grid 0, cph is non-FEP), wired in md.cpp
  gated on `useGpuPme` (no-op for `-pme cpu`, which reads `mdatoms->chargeA` directly).
- **Potential D2H:** moved out of the `!useGpuFBufferOps` guard so it also fires on the resident path
  (host still needs the per-atom potential to drive the λ ODE while forces stay on device); synced by the
  existing `gpu_wait_finish_task` over the whole local stream.
- **Guards:** `decidegpuusage` allows `useGpuUpdate` for cph only single-GPU (no DD, no separate PME
  rank); `decidesimulationworkload` keeps buffer ops off for cph ONLY when GPU update is off, so the
  validated classic (non-resident) GPU path is unchanged. ‼️ GPU update needs v-rescale/no thermostat
  (NOT Nose-Hoover) — the validation tpr was regrompp'd with `tcoupl=v-rescale` and a heavy
  `lambda-particle-mass` (stable fresh start).
- **Validated (RTX 3080, 88k all-atom PME cph):** R0 (`-pme cpu` resident) single-point dV/dλ vs CPU
  **1.4e-5**, 1000-step λ vs classic-GPU identical through step ~50 → FP divergence 2.5e-3@1000; R0b
  (`-pme gpu`, FULL all-GPU resident) single-point **6.0e-5** (= the L1.2 PME floor), 1000-step FP
  divergence 4.9e-3@1000, λ actively titrating (range [-0.04, 1.06]).
- **Still O(Natoms) per step** (correctness-first): potential D2H + NB charge H2D + PME coeff H2D. The
  device group-reduce (L2.2) and device charge-scatter (L2.3) that make transfers O(Ngroups) are next.

### ✅ Clean-build fixes + own-FFTW re-validation (2026-07-14)
A from-scratch CPU configure/build with `-DGMX_BUILD_OWN_FFTW=ON` (fresh `build-own-fftw`,
gcc 16.1.1, downloads+static-links FFTW 3.3.10) exposed **two latent build breaks** that the
incremental `build-cpu` had been masking (the affected TUs were never recompiled after the L3
header change). Both are introduced by the L3 GPU work and are inert for GPU (CUDA) builds:
- **Compile (`ewald/pme.h`):** the `GPU_FUNC_QUALIFIER` PME-GPU stubs take
  `gmx::ArrayRef<real> electrostaticPotential` **by value**, but the header only forward-declared
  `ArrayRef`. With `GMX_GPU=OFF` those stubs become `gmx_unused static {…}` *definitions*, where a
  by-value parameter of incomplete type is illegal → `'electrostaticPotential' has incomplete type`
  (in `taskassignment/*.cpp`, `modularsimulator.cpp`). Fix: `#include "gromacs/utility/arrayref.h"`,
  drop the now-redundant forward decl.
- **Link (`ewald/pme.cpp`):** L3.1 `gmx_pme_reinit_charges_gpu` called the GPU-only internal
  `pme_gpu_realloc_and_copy_input_coefficients` (no CPU stub; its TU `pme_gpu_internal.cpp` is not
  compiled off-GPU) under `#if !GMX_DOUBLE` → undefined reference in a CPU build. Fix: guard with
  `#if GMX_GPU && !GMX_DOUBLE` (dead code on CPU — `pme->gpu` is always null there, early-returns).
- **Validated (build-own-fftw vs 2021 fork, single point):** M0a PME 46/46 groups max_abs 1.9e-3 /
  single-prec floor (rel 1.3e-5); M1 RF ref-kernel max_abs 5.0e-5, SIMD max_abs 1.0e-4 vs fork and
  6.0e-5 vs ref — i.e. bit-for-bit the documented `build-cpu` figures. Parity with the fork stands;
  the two fixes do not perturb CPU numerics (one compiled out on CPU, one a header include).

### 🐛→✅ BUG FIX: missing 1-4 pair contribution to the per-atom potential (2026-08-05)

**Symptom** (reported in `CPH_AA_BUG_REPRODUCER.md`): an all-atom CHARMM36m system
(90 279 atoms, PME, 45 titratable sites / 54 λ coordinates, 90 BUF) aborted on the first
λ update in the 2026 port —

```
constant_ph.cpp: Assertion failed: (lambdaCoordinate.x < 1.15 && lambdaCoordinate.x > -0.15)
Lambda coordinate left the range for which it has been parametrised.
```

while the **same** files ran fine in the 2021 fork, and ran in the port with λ frozen
(`lambda-dynamics-calibration = yes`). Not the GPU, not the atom order, not the initial λ,
not the multistate His path, not the timestep — all ruled out in the report.

**Root cause.** The port never implemented the fork's hunk in
**`src/gromacs/listed_forces/pairs.cpp`** (`gromacs/listed_forces/pairs.cpp|39|8` in
`_portref/00_stat.txt`). The **1-4 pair (`[ pairs ]`) interactions contribute to the per-atom
electrostatic potential** that drives dV/dλ, exactly as the non-bonded kernel does, and that
contribution was simply absent: `src/gromacs/listed_forces/` was still pristine 2026.1.
For all-atom force fields the missing term is huge — up to ~950 kJ/mol/e for ARGT,
**enough to flip the sign of dV/dλ** on 12 of 54 coordinates — so λ was driven out of its
parametrised range on the first update. This is the [[feedback_get_the_logic]] failure mode
again: the potential must ride *every* place the electrostatic energy is computed, not just
the NB kernel and PME.

**Why the CG campaign never hit it:** Martini topologies have **no `[ pairs ]` section at all**
(verified across the full_size campaign `.itp`/`.top` set), so the code path is never entered
and the 480-window CG results are bit-identical. The AA↔fork dV/dλ comparison had never been
run — M0a/M1 are Martini, and the 88k all-atom PME gates (L1.2/L3) compared port-GPU against
port-CPU, so a term missing from *both* cancelled out.

**Fix.** Port the fork's hunk, adapted to 2026 APIs:
- `evaluate_single` gains an optional `real* coulombValue` out-param returning the tabulated
  Coulomb value **without** the charge-product prefactor (`velec == qq * coulombValue`).
- `do_pairs_general` accumulates `potential[a] += prefactor * q_partner * coulombValue` for
  atoms with `constantPH->isLambdaAtom()`, under `#pragma omp atomic` (the potential buffer is
  shared across the listed-forces threads, unlike the force buffers). Unlike the fork, the
  prefactor/partner-charge pair is set per-ftype alongside `qq`, so `LJC14Q`/`LJCNBPairs`
  (which take their charges from `iparams`, not `mdatoms->chargeA`) are also correct; for
  `LJ14` it reduces to the fork's `epsfac * fudgeQQ * chargeA[partner]`.
- **`do_pairs` must not take the fast path when the potential is being accumulated.** That
  path computes forces only and never evaluates the Coulomb table. It is taken whenever
  `!computeVirial && !computeEnergy`, i.e. on *most* steps (`nstcalcenergy`), so without this
  guard the 1-4 term would appear only on energy steps — inconsistent forces, not just a
  constant offset. Guard: `&& fr->electrostaticPotential.empty()`.
- **New guard (readir.cpp):** cpHMD + free-energy perturbation is now a hard grompp **error**.
  The fork silently suppressed the perturbed 1-4 path when cph was active
  (`bFreeEnergy && fr->electrostaticPotential.empty()`); silently changing FEP results is a
  trap, so the combination is refused instead. Also in `README.md`.

**Validation (local `build-cpu` vs the 2021 fork oracle, same 90 279-atom AA system; each
side grompps its own tpr).** New gate **M6** — the first AA all-atom dV/dλ comparison
against the fork:

| check | result |
|---|---|
| M6a single point, PME, 54 coords | max abs diff **5.0e-4**, worst rel **2.1e-5** |
| M6b single point, Reaction-Field | max abs diff **4.9e-3**, worst rel **5.0e-6** |
| M6c single point at a *displaced* geometry + all 54 λ off-centre (multistate HSPT off equal splits) | worst rel **2.7e-6** |
| M6d 60-step λ trajectory, fixed λ seeds, `tcoupl=no pcoupl=no` | λ matches fork to **5.5e-5** = the fork's xvg output precision; atomic coords identical to `.gro` precision |
| M6e the reproducer itself: 1500 steps, production mdp (dt 0.002, c-rescale, all 4 HSPT) | **no assertion**; λ evolves over [-0.116, 1.074], i.e. inside the ±0.15 tolerance, same order as the fork's own -0.065 excursion |

Before the fix the same single point was off by a **per-residue-type** offset —
LYST -14, ASPT -145, GLUT -164, ARGT **-951**, HSPT +68/+150/+99, BUF exactly 0 (one atom ⇒ no
1-4 pairs) — and, diagnostically, **identical for RF and PME**, which is what pinned the fault
on a coulombtype-independent term (the 1-4 pairs) rather than on the NB kernel or the
reciprocal path. The port's SIMD and plain-C kernels agreed with each other to 9e-4 throughout,
which ruled the kernels out early.

**Note on the 60-step λ divergence with coupling ON** (0.024 at step 60, seen before M6d): that
is **not** cph. It survives pinning `ld-seed`, and disappears completely when
`pcoupl`/`tcoupl` are switched off — the 2021 and 2026 c-rescale/verlet-buffer implementations
differ (this system runs at ~5900 bar), so the *atomic* trajectories separate and λ follows.
Any port↔fork λ-trajectory comparison must therefore run with coupling off.

**Files:** `src/gromacs/listed_forces/pairs.cpp`, `src/gromacs/gmxpreprocess/readir.cpp`,
`README.md`. Non-cph runs are untouched: with lambda dynamics off
`fr->electrostaticPotential` is empty ⇒ `potential == nullptr`, the fast path guard is
`true`, and `evaluate_single` gets `nullptr`.

**Follow-up on the GPU path (same day).** With the `pairs.cpp` fix in place the CPU run was
clean but `-nb gpu -pme gpu` (with `-bonded auto`, which offloads listed forces once NB *and*
PME are on the GPU) still hit the assertion at step 0. Cause: the **GPU bonded kernels compute
the 1-4 pairs on the device and do not accumulate the per-atom potential**, so offloading them
silently drops the term again. Fix: `inputSupportsListedForcesGpu()` now reports
`lambda_dynamics` as an unsupported input, so `-bonded auto` keeps listed forces on the CPU and
an explicit `-bonded gpu` fails fast (`InconsistentInputError`) instead of giving wrong dV/dλ.

Bisection at `nsteps 0` on the reproducer's own tpr (L40S, `-bonded cpu` forced), each vs the
CPU reference: `-nb gpu` **7.1e-5**, `-nb gpu -pme gpu` **1.6e-4**, `-nb gpu -pme gpu -update
gpu` **1.5e-4** worst relative (worst cases are small-magnitude HSPT coordinates; ≤0.017
absolute) — i.e. the documented single-precision cuFFT floor, so the GPU potential path itself
was never at fault.

**GPU-resident path and the ±0.15 boundary (closing the reproducer).** After the two fixes, a
step-bisection on the reproducer's own tpr (L40S, 400 steps each, `-bonded cpu`) gives:
`-nb cpu` clean, `-nb gpu -pme cpu` clean, `-nb gpu -pme gpu` clean, `-nb gpu -pme gpu -update
gpu` **trips the assertion at step ~397**. That is **not** an L3 bug: the λ deviation from the
CPU run starts at the FP floor (0 at step 0, 6e-6 at step 1) and grows exponentially
(1.6e-3@10, 3.6e-2@50, 0.17@100 …) — Lyapunov divergence, identical in character to the
`-pme gpu` (update-cpu) run, which starts at 1e-6 and reaches 0.74 by step 397 without tripping.

The real finding is that **this AA setup runs right at the assertion boundary.** Over the same
400 steps from the same start, λ excursions are:

| run | λ range over 400 steps |
|---|---|
| 2021 fork (CPU) | **[-0.1392, 1.0921]** |
| port (CPU) | **[-0.1408, 1.0994]** |
| port `-nb gpu -pme gpu` | [-0.1049, 1.0952] |
| port GPU-resident | [-0.1472, 1.0852] → trips |

Fork and port are statistically identical, i.e. the port now reproduces the fork's λ dynamics on
this system — and **both are one FP perturbation away from -0.15**. Whether a given run survives
is decided by rounding, which is exactly what the GPU-resident case demonstrates. This is a
**run-parameter (science) matter, not a port defect**: with `lambda-particle-mass = 5.0` and
`barrier = 7.5` this system drives λ ~0.14 past the well. Remedies are the user's call — heavier
λ particle mass, larger barrier, smaller `dt`, longer fixed-λ equilibration, or a review of the
dvdl calibration for these residue types.

To make that diagnosable, the bare `GMX_RELEASE_ASSERT` in `updateLambdas` is replaced by a
`gmx_fatal` naming the coordinate, its λ value and the step, and pointing at the knobs — same
abort semantics, but it now distinguishes "one group is running away" from "the whole setup is
marginal".

### 🛡️ Guard: constant pH on non-CUDA GPU backends (2026-08-06)

The GPU per-atom-potential kernels are **CUDA only** (`nbnxm/cuda/nbnxm_cuda_kernel.cuh`,
`ewald/pme_gather.cu`); `grep` finds no potential accumulation anywhere under `nbnxm/sycl/`.
The buffer and its plumbing, however, are backend-agnostic (`gpu_types_common.h`,
`nbnxm_gpu_data_mgmt.cpp` sets `computePotential` regardless of backend), so a **SYCL or HIP**
build running cph with `-nb gpu` would allocate the buffer, clear it, copy it back as zeros and
lose the **entire real-space term** of dV/dλ — silently, the characteristic cph failure mode.

Guard added in `canUseGpusForNonbonded()`: `lambda_dynamics && !GMX_GPU_CUDA` is now an
unsupported-input reason. That one function feeds **both** `decideWhetherToUseGpusForNonbonded`
and its thread-MPI twin via `canUseNonbondedOnGpu`, so:
- `-nb auto` (the default) → falls back to CPU non-bonded and logs the reason. Correct results
  by default, which matters more here than the lost performance.
- `-nb gpu` → `InconsistentInputError` ("not supported for these simulation settings").
- PME needs no separate guard: `canUseGpusForPme` already requires non-bonded on the GPU.
- **CUDA builds are unaffected** (`GMX_GPU_CUDA` = 1 ⇒ the reason is never appended), and the
  message is only printed under `if (GMX_GPU && ...)` in `runner.cpp`, so CPU-only builds stay
  silent.

Tested: on the CPU-only `build-cpu` the true branch is the one taken (non-CUDA build), and cph
still runs with dV/dλ unchanged vs the fork (worst rel 1.5e-5) with no spurious log message.
The CUDA build re-verified unaffected (see below). **The SYCL/HIP branch itself is untested —
no such build exists here**; only the condition and the fallback behaviour were exercised.

### 🐞 Bug: PME on the GPU used a stale reciprocal potential / stale charges (2026-08-06)

Reported symptom: **`-update gpu` together with `-pme gpu` is broken.** It is worse than that —
`-pme gpu` was wrong on *both* GPU paths, and for two independent reasons. Both are the classic
cph failure mode (a silently stale/never-refreshed buffer, not a physics error), and both were
hidden by the fact that **every earlier GPU gate was a single point (`nsteps 0`) or an
energy/virial step**, which is exactly where neither bug fires.

The pivot is `StepWorkload::useGpuFBufferOps = useGpuFBufferOpsWhenAllowed && !computeVirial`
(`decidesimulationworkload.cpp:314`) and, from it,
`useGpuPmeFReduction = computeSlowForces && useGpuFBufferOps && haveGpuPmeOnPpRank()`. On a
virial/energy step buffer ops go **off**, so the PME forces come back to the host and everything
works. On an ordinary step (`nstcalcenergy` = 100 in production ⇒ 99 steps out of 100) they stay
on the device — and that is where the potential path fell apart.

**Bug 1 — the per-atom reciprocal potential D2H was never launched on the resident path.**
Its copy rode inside `pme_gpu_copy_output_forces()`, which `pme_gpu_gather()` calls only in the
`else` branch of `if (settings.useGpuForceReduction)`. With `-pme gpu -update gpu` that branch is
never taken on a non-virial step, so `staging.h_potentials` kept whatever the last virial step had
left in it, while `pme_gpu_reduce_outputs()` went on adding it to `fr->electrostaticPotential`
every step. A second, matching hole: `pme_gpu_wait_finish_task()` synchronised the PME stream only
`if (!useGpuForceReduction || computeEnergyAndVirial)`, so even a launched copy would not have been
waited for. Fix: split the potential staging out into `pme_gpu_copy_output_potentials()` and launch
it from `pme_gpu_gather()` unconditionally (it is a host-consumed output every step, unlike the
forces), and add `settings.computeElectrostaticPotential` to the stream-sync condition.

**Bug 2 — the PME device charge buffer was never refreshed on the classic GPU path.**
`d_coefficients` is only written by `gmx_pme_reinit_atoms()`, i.e. at startup and at each DD
repartition — nothing else. The L3.1 per-step refresh (`gmx_pme_reinit_charges_gpu`, commit
`e300526`) was placed **inside** `if (useGpuForUpdate)`, so `-nb gpu -pme gpu` *without*
`-update gpu` ran PME reciprocal space — forces **and** the potential that drives dV/dλ — with the
λ-charges frozen at their step-0 values for the whole run. Fix: the refresh is unconditional on
`simulationWork.useGpuPme` (no-op for PME on CPU, which reads `mdatoms->chargeA` directly).

**Bug 3 (latent) — the NB per-atom potential D2H was launched but never waited for.**
`gpu_launch_cpyback()` correctly fires the potential D2H on both paths (commit `0c8244f`), but
`gpu_try_finish_task()`'s `haveResultToWaitFor` only covered `!useGpuFBufferOps` and
energy/virial steps, so on the resident path nothing synchronised the local NB stream before
md.cpp read the host buffer. `-pme cpu` masks it (CPU PME is slow enough that the copy always
lands); `-pme gpu` removes that cushion. Fix: `|| nb->atdat->computePotential` — the potential is
a host-consumed result whenever it is computed. Untriggered in the runs below, but a real race.

**Validation** — 90k all-atom CHARMM36m PME, 54 λ coords, 40 steps, `nstcalcenergy = 100` (so
only step 0 is an energy step), `tcoupl = no`/`pcoupl = no`, `nstlist = 20`, RTX 3080 (b32_128_gpu).
Max |Δλ| over all 54 coordinates vs the same build's `-nb cpu -pme cpu` reference, before → after:

| mdrun flags | step 1 | step 10 | step 39 |
|---|---|---|---|
| `-nb gpu -pme gpu` (classic) | 7e-6 → **0** | 1.4e-3 → **1e-6** | **2.8e-2** → **4e-6** |
| `-nb gpu -pme cpu -update gpu` | 0 → 0 | 1e-6 → 1e-6 | 5e-6 → **3e-6** |
| `-nb gpu -pme gpu -update gpu` (resident) | 7e-6 → **0** | 1.4e-3 → **1e-6** | **2.8e-2** → **4e-6** |

1e-6 is the `GMX_CPH_DUMP_LAMBDAS` output floor (5 decimals). The two columns isolate the two
bugs cleanly: for the *classic* row the only behavioural change is bug 2's fix (bug 1 needs
`useGpuForceReduction`), and for the *resident* row the only changes are bugs 1+3 (its charge
refresh already ran). `-pme cpu -update gpu` was and remains correct, which is why the port log
did not catch this earlier. Repro: `cphfix_job.sh` + `cphfix/md_fix.mdp` in
`aurum2:~/work/Misha/CG/full_size_1d/aa_cph/pure_w113_fixlam/`.

**Lesson (same family as the M6 1-4-pair bug).** Every GPU gate so far was `nsteps 0` or compared
GPU-vs-GPU. A single point is a *virial step*, and on a virial step the GPU-resident path silently
degrades to the classic path — so it cannot test residency at all. **Any future GPU gate must run
multiple steps with `nstcalcenergy` larger than the run length**, so that the ordinary-step
schedule is the thing under test.

### ✅ L5 (HIP): constant-pH GPU port to AMD / LUMI (2026-08-29, branch `hip-port`)

Ported the three cph GPU device kernels from CUDA to the HIP backend so the method runs
GPU-accelerated on LUMI-G (AMD MI250X, `gfx90a`) with VkFFT for PME. Built via EasyBuild
(`cpeAMD/25.09`, ROCm 6.4.4, clang 19) from the LUMI contrib recipe. ~85 % of the cph GPU code
(buffer mgmt, D2H/H2D staging, MD-loop wiring, PME host plumbing) is backend-shared and already
compiled as HIP; only the three device kernels needed HIP twins. See `HIP_PORT_PLAN.md` for the
per-file map and `CLAUDE.md` ("Building & running on LUMI") for the recipe.

- **Prereq fix (`eaeb03c`).** `gpu_free` freed `NBAtomDataGpu::potential`/`lambdaCharges`
  unconditionally, but they are allocated only when `computePotential`; `new NBAtomDataGpu` leaves
  the raw-pointer members uninitialised, so a non-cph run hipFree'd garbage — silent on CUDA
  (`cudaFree`), fatal on HIP (`hipErrorInvalidValue`) at teardown. Latent on aurum2; surfaced by
  the first LUMI GPU run. Gated the free on `computePotential`. **CUDA↔HIP lesson:** every
  conditionally-allocated `DeviceBuffer` must be null-initialised or its free gated — HIP
  `freeDeviceBuffer` only null-guards, it does not tolerate an uninitialised pointer.
- **H1 (`1f644c3`) — NB per-atom potential** in `nbnxm/hip/nbnxm_hip_kernel_body.h` (one header
  covers all four kernel TUs). Per-pair `V_i += epsFac·qj·coulFunc`, `V_j += qi·coulFunc`, reusing
  the kernel's own cutoff/RF/Ewald energy forms with `qi·qj` factored off; correctness-first
  per-pair `atomicAdd`. **Gate G1:** 46-λ RF Martini single point, `-nb gpu` vs `-nb cpu`, dV/dλ
  max rel **1.3e-6**, all 46 groups non-zero.
- **H2 (`1496430`) — PME reciprocal potential** in `ewald/pme_gather_hip.cpp`: `sumForceComponents`
  accumulates `θx·θy·θz·grid`; after the force reduce, a maskless `__shfl_down` over `atomDataSize`
  lanes (HIP has no `c_fullWarpMask`) collapses to lane `splineIndex==0`, which writes
  `d_potentials`. **Gate G2:** 46-λ PME Martini, `-nb gpu -pme gpu` vs `-nb cpu -pme cpu`, dV/dλ
  max rel **4.5e-5** — the VkFFT single-precision floor (measured; wavefront 32→64 shuffle reduce
  confirmed correct).
- **H3 (`ad7cdfa`) — GPU-resident λ-charge repack** in
  `nbnxm/hip/nbnxm_gpu_buffer_ops_internal_hip.cpp`: the X buffer-op kernel refreshes `xq.w` from
  `lambdaCharges` each step (nullptr for non-cph). **Gate G3:** all-atom CHARMM36 cph
  (constraints=h-bonds → GPU update supported), 100-step `-update gpu` vs `-update cpu` λ
  trajectory, max |Δ| **1.5e-5** (identical to FP). The Martini/PW system cannot test this —
  polarizable water gives triangle constraints, which GPU update refuses on every backend.
- **Guards.** `decidegpuusage.cpp` now admits HIP for cph GPU NB
  (`!(GMX_GPU_CUDA || GMX_GPU_HIP)`); `decidesimulationworkload.cpp` admits cph + PME-on-GPU on HIP.
  SYCL still refused. No CUDA behaviour change (identical predicate on a CUDA build; H1–H3 code is
  in `hip/` files only).
- **Regression.** Non-cph GPU water box unchanged (`-nb gpu -pme gpu -update gpu` ~950 ns/day,
  clean finish); all cph GPU branches are gated on `computePotential` / `gm_charge != nullptr`.

Not yet merged to `master` (branch `hip-port`). Remaining: HIP DD (classic-path multi-GPU) is
untested but backend-shared; a science-level titration gate on LUMI; upstream SYCL if ever needed.
