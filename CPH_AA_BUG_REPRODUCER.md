# Reproducer: 2026 port kills lambda dynamics at step 0 on an all-atom CHARMM36m system

> ## ✅ RESOLVED — 2026-08-05
>
> **Root cause:** the port never implemented the fork's hunk in
> `src/gromacs/listed_forces/pairs.cpp`. The **1-4 pair (`[ pairs ]`) interactions contribute
> to the per-atom electrostatic potential** that drives dV/dλ, and that contribution was
> missing entirely (`src/gromacs/listed_forces/` was still pristine 2026.1). For all-atom
> force fields the missing term reaches ~950 kJ/mol/e (ARGT) — **enough to flip the sign of
> dV/dλ on 12 of the 54 coordinates** — so λ was driven out of [-0.15, 1.15] on the first
> update. Martini has no `[ pairs ]` section at all, which is why the 480-window CG campaign
> never hit it, and why the existing gates missed it (M0a/M1 are Martini; the 88k all-atom
> gates compared port-GPU against port-CPU, so a term missing from both cancelled).
>
> **Diagnostic that pinned it:** the port↔fork dV/dλ offset was **identical for
> Reaction-Field and for PME** (LYST -14, ASPT -145, GLUT -164, ARGT -951, HSPT +68/+150/+99,
> BUF exactly 0) — a coulombtype-independent, per-residue-type term, i.e. not the NB kernel
> and not the reciprocal path.
>
> **Fixed in** `src/gromacs/listed_forces/pairs.cpp` (+ a cpHMD×FEP grompp guard in
> `readir.cpp`). A second instance of the same term was found on the GPU: with NB *and* PME on
> the GPU, `-bonded auto` offloads the listed forces and the GPU bonded kernels do not
> accumulate the potential either — `inputSupportsListedForcesGpu()` now keeps listed forces on
> the CPU for cpHMD. dV/dλ now agrees with the 2021 fork to **2.7e-6** relative on this system.
>
> **Verified on `topol_dyn.tpr` itself** (this directory, the exact tpr from the report):
> `-nb cpu` runs the **full 25 000 steps / 50 ps with no assertion**; `-nb gpu`,
> `-nb gpu -pme gpu` run clean too.
>
> ### ⚠️ Remaining, and it is a run-parameter matter, not a port bug
>
> With the force path correct, **this setup runs right at the ±0.15 boundary.** Over the same
> 400 steps from the same start: fork λ ∈ **[-0.1392, 1.0921]**, port λ ∈ **[-0.1408, 1.0994]** —
> statistically identical, and both within rounding distance of -0.15. Consequently the
> **GPU-resident** path (`-nb gpu -pme gpu -update gpu`) still trips the check, at step ~397; its
> deviation from the CPU run starts at the FP floor (6e-6 at step 1) and grows exponentially, so
> that is Lyapunov divergence deciding which run touches the wall first, not a defect in L3.
>
> So the note at the bottom of this report — "the port arrives there in one step rather than
> drifting there over picoseconds" — was the bug and is fixed; but λ *does* legitimately drift to
> ≈-0.14 in both codes here. With `lambda-particle-mass = 5.0` and `barrier = 7.5` this system
> pushes λ ~0.14 past the well. Choices, all yours: heavier λ particle mass, larger barrier,
> smaller `dt`, longer fixed-λ equilibration, or a review of the dvdl calibration for these
> residue types. The assertion now reports **which** coordinate, its λ value and the step, so
> "one group running away" is distinguishable from "the whole setup is marginal".
>
> Full validation table (gate M6) in `PORT_LOG.md`, entry "BUG FIX: missing 1-4 pair contribution
> to the per-atom potential".
>
> Everything below is the original report, kept as-is.


The 2021 constant-pH fork runs this system fine. The 2026 port aborts on the **first**
lambda update with

```
src/gromacs/applied_forces/constant_ph/constant_ph.cpp (line 1569)
Function: ConstantPH::updateLambdas(int64_t)::<lambda()>
Assertion failed:
  Condition: (lambdaCoordinate.x < 1.15 && lambdaCoordinate.x > -0.15)
  Lambda coordinate left the range for which it has been parametrised.
```

## Where the files are

```
aurum2:/home/yesylevskyy/work/Misha/CG/full_size_1d/aa_cph/pure_w113_fixlam/
```

Everything below lives in that one directory. Same coordinates, topology, index and
lambda block throughout — only the mdp and the binary differ.

| file | what it is | result |
|---|---|---|
| `topol_2021.tpr` | dynamic lambda, grompp'd with the **2021 fork** | **runs**: 10 ps, 0 crashes, 18.0 ns/day on 16 CPU threads, lambda evolves |
| `topol_dyn.tpr` | dynamic lambda, grompp'd with the **2026 port** | **dies at step 0** |
| `topol_nohis.tpr` | same, all 4 multistate HSPT removed (single-state groups only) | dies at step 0 |
| `topol_dt02.tpr` | same as `topol_dyn` but `dt = 0.0002` (10x smaller) | dies at step 0 |
| `topol.tpr` | **lambda held fixed** (`lambda-dynamics-calibration = yes`), 2026 port | **runs**: 94.7 ps stable |
| `out_cpu.cpt` | the relaxed state from that fixed-lambda run | used as `-t` for the dynamic tprs |
| `conf.gro`, `topol.top`, `index.ndx`, `cph.part`, `protein.itp` | the system | — |
| `md_dyn.mdp`, `md_nohis.mdp`, `md_dt02.mdp`, `md_2021.mdp`, `md.mdp` | the mdps for the above | — |

## Binaries

```bash
# 2026 port (fails)
GH=/home/yesylevskyy/install/gromacs-2026-cph
export LD_LIBRARY_PATH=$GH/build-sys/lib:$LD_LIBRARY_PATH
$GH/build-sys/bin/gmx mdrun -s topol_dyn.tpr -ntmpi 1 -ntomp 16 -nb cpu

# 2021 fork (works)
F=/uochb/soft/users/yesylevskyy/d/gromacs-cph-new
export LD_LIBRARY_PATH=$F/lib64:$F/lib:$LD_LIBRARY_PATH
$F/bin/gmx mdrun -s topol_2021.tpr -ntmpi 1 -ntomp 16 -nb cpu
```

A tpr must be grompp'd by the binary that will run it — the tpr versions differ.
Both grompp runs are clean, **zero** non-zero-total-charge warnings.

## The system

All-atom CHARMM36m, 90 279 atoms, PME, `dt = 0.002`, `constraints = h-bonds`,
`charmm36-mar2019-cphmd.ff`. Backmapped from a coarse-grained truncated umbrella window.

- **45 titratable sites**: ASPT 6, GLUT 11, **HSPT 4 (3-state, multistate constraints)**,
  ARGT 12, LYST 12 — plus 90 BUF buffer particles. 54 lambda coordinates in total
  (each HSPT occupies three), reported as 14 `cphmd` output files.
- `lambda-dynamics-charge-constraints = yes`, `multistate-constraints = yes`, pH 7.2.
- Group-type parameters (charges, reference pKa, dvdl coefficients) are the CHARMM36m
  values taken verbatim from a working production setup,
  `aurum2:/home/yesylevskyy/work/Misha/Rotate/N_only_rot_lip_restr_com/cph.part`,
  with the HSPT reference pKa set shifted +0.2 (7.20/6.73/7.12) so one state equals the
  simulation pH, as the multisite convention requires.
- Neutral at **lambda = 0**: protein +28 (fully protonated reference) − 230 POPG
  + 223 Na+ − 21 Cl− = 0. All sites start at lambda = 0.5 (HSPT at 0.5/0.25/0.25, sum 1),
  and the buffer carries the 22.5 e that removes, at lambda_BUF = 0.5 + 22.5/90 = 0.75.

## What has already been ruled out

- **not the GPU** — CPU and CUDA builds of the port fail identically, same assertion, same
  step. Independently, the port runs coarse-grained cph fine on GPU: 1552 ns/day, lambda
  read-back clean.
- **not the atom order** — every one of the 45 lambda index groups was verified to hold
  exactly the atoms its group-type charge vector expects, in that order.
- **not the initial values** — fails from lambda = 0.5 (middle of the well) just as it does
  from 0/1.
- **not a strained environment** — fails when continued from 94.7 ps of stable
  fixed-lambda relaxation.
- **not the multistate His path** — fails with all HSPT removed, leaving only single-state
  groups with 6-coefficient dvdl fits.
- **not integration stability / timestep** — fails at `dt = 0.0002` too. At one tenth the
  timestep the lambda step would be ~100x smaller for the same force, so leaving a range of
  width 1.3 means the first update is invalid, not merely too large.
- **not the system** — the same files run in the 2021 fork, and run with lambda frozen in
  the port itself.

Conclusion: the fault is in how the 2026 port computes or applies the lambda force /
dV/dlambda for these parameter sets. It is on the first update, before any dynamics.

## Why the coarse-grained campaign never hit it

The 480-window CG cph campaign runs on this port without trouble. Its dvdl coefficients are
Martini-recalibrated and far smaller — 6 terms of order 1e1–1e2 — whereas the AA sets reach
1e2–1e3 for the single-state groups and ±1e5 across the 45-term multistate His fit. Whatever
is wrong scales with those magnitudes or with the polynomial layout.

## Useful next step

The 2021 fork's own known-good input is at
`aurum2:/home/yesylevskyy/work/Misha/Rotate/N_only_rot_lip_restr_com/` (same protein's
N-domain, 3-state HSPT, ran hundreds of ns). Re-grompp it with the port and run: it isolates
the port from this particular assembly using an input that has never been in doubt.

Also worth noting: the 2021 fork's own lambda trace reaches −0.065, outside [0,1] but inside
its tolerance. Policing ±0.15 is not itself wrong — the port arrives there in one step
rather than drifting there over picoseconds.

## Practical consequence

Measured: **18.0 ns/day** on 16 CPU threads (2021 fork). The fork does report CUDA support,
and the CG benchmark gives ~2.3x for GPU over CPU with the same binary, so expect roughly
40–50 ns/day on one L40S. Production at all-atom therefore costs ~10–12 GPU-days per 500 ns
window, i.e. ~3500 GPU-days for 320 windows — not reachable, independent of this bug.
