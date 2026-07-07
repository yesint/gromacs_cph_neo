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
on the fast SIMD kernels instead of the plain-C reference. **Opt-in via `GMX_CPH_SIMD=1`**; the
validated default stays plain-C 4x4 (risk-free). Files touched (all under `src/gromacs/nbnxm/`):
- **atomdata.{h,cpp}**: added `computeElectrostaticPotential_` flag (+ getter/setter). This is the
  runtime signal that gates the SIMD potential work — set = `inputrec.lambda_dynamics` in
  `nbnxm_setup.cpp` right after nbat construction. (The potential *buffer* resize stays keyed on the
  compile macro so the reference kernel is untouched; the SIMD kernel gates on the flag, so non-cph
  SIMD runs skip the work.)
- **nbnxm_setup.cpp** `pickNbnxnKernelCpu`: cph forces plain-C 4x4 **only when `GMX_CPH_SIMD` is unset**;
  with it set, cph uses the normal SIMD selection.
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

**How to run cph on SIMD:** `GMX_CPH_SIMD=1 LD_LIBRARY_PATH=.../build-cpu/lib gmx mdrun ...` (auto 4xN;
force layout with `GMX_NBNXN_SIMD_4XN` / `GMX_NBNXN_SIMD_2XNN`). Without `GMX_CPH_SIMD`, still plain-C 4x4.

## THE PORT IS FULLY COMPLETE — remaining items are cosmetic/optional
- Cosmetic: edr block name shows 'id' not 'Constant pH data' in gmx dump (cphmd matches by numeric id).
- Multi-rank DD cph (real commrec) — not needed for the single-rank campaign.
- M3–M5 (titration/pKa vs fork; full two-domain window; refdata regression test) — science validation,
  not port work.
