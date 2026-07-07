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

## Status
- WP0 ✓  | WP1 module files copied + CMake registered (add_subdirectory constant_ph); API fixes pending.
- WP2/3/4/5a pending. Reaching a compiling module needs WP2 types first (compiler-driven).
