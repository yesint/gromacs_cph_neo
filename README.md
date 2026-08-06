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
- The per-atom electrostatic potential (the dV/dλ driver) accumulated in the
  plain-C reference **and** the SIMD non-bonded kernels, so constant pH runs on the
  fast SIMD path (~7.7× the plain-C kernel).
- **Multi-rank domain decomposition**, including PME on the PP ranks: the per-atom
  potential is halo-exchanged with the force reduction and the reciprocal part is
  redistributed with the PME force redistribution.
- **GPU (CUDA)**: the non-bonded kernel and the PME reciprocal-space gather both
  accumulate the per-atom potential on the device, and the fully GPU-resident loop
  (`-nb gpu -pme gpu -update gpu`) is supported on a single GPU, with only the small
  λ ODE on the host.
- Force/energy path validated against the original 2021 fork: per-group dV/dλ agrees
  to ~1e-4 or better in single precision, on Martini/reaction-field, Martini/PME,
  under domain decomposition, and on an all-atom CHARMM36m PME system with charge and
  multistate constraints (worst relative deviation ~3e-6 there).

## Limitations

1. **GPU support is CUDA only.** The per-atom-potential kernels exist for CUDA
   (non-bonded and PME gather). **SYCL and HIP builds have no such kernels**, and
   nothing currently stops you from running there — `-nb gpu` on a SYCL/HIP build
   would silently give dV/dλ = 0. On those builds run `mdrun -nb cpu`. GPU
   *emulation* (`GMX_EMULATE_GPU`) does fail fast.

2. **The GPU-resident update (`-update gpu`) needs a single GPU**: no domain
   decomposition and no separate PME rank. DD or separate-PME runs fall back to the
   classic (non-resident) GPU path automatically. The stock GPU-update restrictions
   also apply — in particular **no Nose-Hoover**; use `tcoupl = v-rescale` or `no`.

3. **Separate PME ranks are not supported** (`mdrun -npme 0`; PME runs on the PP
   ranks). The reciprocal-space potential is computed on the PME rank and is not sent
   back to the PP ranks, so this **fails fast** rather than dropping the reciprocal
   part of dV/dλ.

4. **Listed forces always run on the CPU.** The GPU bonded kernels do not accumulate
   the per-atom potential, so 1-4 pair interactions would lose their contribution to
   dV/dλ. With `-bonded auto` the listed forces stay on the CPU automatically; an
   explicit `-bonded gpu` fails fast. Costs a little performance on GPU runs.

5. **edr λ-output requires `nstenergy > 0`, aligned to `lambda-dynamics-update-nst`.**
   The λ block is written only on energy-file frames. With `nstenergy = 0` the λ
   frames carry no standard energy terms (`nre = 0`) and **cannot be read back**
   by `gmx cphmd` / `gmx check` / `gmx energy` (`"Energy header magic number
   mismatch"`). Set `nstenergy = lambda-dynamics-update-nst` so λ lands in
   readable frames.

6. **Leap-frog (MD) integrator only.** Other integrators set the constant-pH
   charges but do not perform λ dynamics.

7. **λ thermostat: no Langevin** (fails a runtime assertion); use a supported
   λ thermostat.

8. **Constant pH cannot be combined with free-energy perturbation.** `lambda-dynamics = yes`
   together with `free-energy != no` is refused by `grompp`. The 1-4 pair kernel cannot
   produce both the free-energy derivative and the per-atom electrostatic potential that
   λ dynamics needs, so allowing the combination would silently change the free-energy
   results.

9. **`.tpr` files are not portable** between the original 2021 fork and this 2026
   port (tpx version gap). Always `grompp` with the same build you run `mdrun`
   with.

10. **SIMD is the default cph kernel** (AVX2). Set `GMX_CPH_NO_SIMD=1` to fall back
    to the slower plain-C reference kernel (debugging only).

11. **On the GPU path the per-step host↔device traffic is still O(N_atoms)** (potential
    copied back, λ charges pushed out) rather than O(N_groups). Correct, but not yet
    the intended performance.

12. **Science-level validation incomplete.** The force/energy path, checkpointing,
    and `gmx cphmd` are validated against the fork, but higher-level checks
    (titration / pKa vs the fork, a full production window, refdata regression)
    have not been done.

## Practical note on the λ range

λ coordinates are policed to `[-0.15, 1.15]`, the range the dV/dλ fits are parametrised
for, and leaving it aborts the run naming the offending coordinate, its λ value and the
step. λ does legitimately stray somewhat outside `[0, 1]`; if a run aborts, treat it as
the λ dynamics being driven too hard rather than as a solver failure, and look at
`lambda-dynamics-lambda-particle-mass`, the group `barrier`, the dvdl coefficients for
that group type, `dt`, and how well the system was equilibrated at fixed λ. All-atom
setups sit closer to this boundary than coarse-grained ones.
