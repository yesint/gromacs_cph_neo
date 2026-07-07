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
- **WP4 (do_md hooks)** — NEXT. Thread ConstantPH* into LegacySimulator::do_md; hooks: setLambdaCharges
  before do_force, updateLambdas after update (F_EPOT/F_EKIN/lambda_therm_integral), output gate.
  Needs ewcCONSTANTPH + creating ConstantPH in runner/simulatorbuilder. 2026 do_md heavily refactored.
- **WP5a (force path)** + **M1** after WP4.
