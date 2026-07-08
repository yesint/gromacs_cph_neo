# GROMACS 2026.1 — Constant-pH (λ-dynamics)

This is GROMACS **2026.1** with **constant-pH (λ-dynamics)** molecular dynamics.
It ports the constant-pH implementation of Aho, Buslaev *et al.* — originally
developed as a fork of GROMACS 2021 — onto a pristine GROMACS 2026.1 baseline, so
that constant-pH simulations run on a current GROMACS.

The method, its theory, and the input/output and setup workflow are **unchanged**
from the original project and are documented there. Prepare and run constant-pH
systems exactly as described upstream, subject to the limitations below.

## Upstream constant-pH resources

- **Original constant-pH GROMACS code + manual:** https://gitlab.com/gromacs-constantph/constantph
- **Tutorials / example input files:** https://gitlab.com/gromacs-constantph/tutorials
- **Constant-pH force fields** (required to prepare titratable systems): https://gitlab.com/gromacs-constantph/force-fields

If you use constant-pH MD, cite:

- Aho N., Buslaev P., Jansen A., Bauer P., Groenhof G., Hess B.
  *Scalable Constant pH Molecular Dynamics in GROMACS.* J. Chem. Theory Comput.
  **18**(10), 6148–6160 (2022).
- Buslaev P., Aho N., Jansen A., Bauer P., Groenhof G., Hess B.
  *Best Practices in Constant pH MD Simulations: Accuracy and Sampling.*
  J. Chem. Theory Comput. **18**(10), 6134–6147 (2022).

## What the 2026.1 port adds

- The constant-pH module carried onto GROMACS 2026.1 (`grompp`, MD, `gmx cphmd`).
- λ coordinates/velocities written to and restored from checkpoints, so `-cpi`
  continues a λ trajectory unbroken.
- The per-atom electrostatic potential (the dV/dλ driver) accumulated in both the
  plain-C reference and the SIMD non-bonded kernels, so constant pH runs on the
  fast SIMD path.
- Force/energy path validated against the original 2021 fork (per-group dV/dλ
  agrees to ~1e-4, single precision).

## Limitations

1. **CPU non-bonded only — no GPU.** A `lambda_dynamics` run with GPU (or
   GPU-emulation) non-bonded kernels **fails at setup**: the GPU kernels do not
   compute the per-atom electrostatic potential, so a GPU run would silently give
   dV/dλ = 0. Run with `mdrun -nb cpu`. GPU support is **not implemented**.

2. **edr λ-output requires `nstenergy > 0`, aligned to `lambda-dynamics-update-nst`.**
   The λ block is written only on energy-file frames. With `nstenergy = 0` the λ
   frames carry no standard energy terms (`nre = 0`) and **cannot be read back**
   by `gmx cphmd` / `gmx check` / `gmx energy` (`"Energy header magic number
   mismatch"`). Set `nstenergy = lambda-dynamics-update-nst` so λ lands in
   readable frames.

3. **Single rank per simulation.** No multi-rank domain decomposition for
   constant pH; run one rank per simulation (`-ntmpi 1 -nb cpu`).

4. **Leap-frog (MD) integrator only.** Other integrators set the constant-pH
   charges but do not perform λ dynamics.

5. **λ thermostat: no Langevin** (fails a runtime assertion); use a supported
   λ thermostat.

6. **`.tpr` files are not portable** between the original 2021 fork and this 2026
   port (tpx version gap). Always `grompp` with the same build you run `mdrun`
   with.

7. **SIMD is the default cph kernel** (AVX2). Set `GMX_CPH_NO_SIMD=1` to fall back
   to the slower plain-C reference kernel (debugging only).

8. **Science-level validation incomplete.** The force/energy path, checkpointing,
   and `gmx cphmd` are validated against the fork, but higher-level checks
   (titration / pKa vs the fork, a full production window, refdata regression)
   have not been done.

Inherited from upstream: separate PME ranks are not supported with MPI.
