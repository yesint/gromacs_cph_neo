# CLAUDE.md — constant-pH GROMACS 2026 port

Guidance for Claude Code when developing **in this tree**
(`/home/semen/install/gromacs-2026-cph`). This is the development-facing companion
to the other docs already here — read those for the details this file only points to:

| Doc | What it is |
|---|---|
| `PORT_LOG.md` | The running progress log — chronological, per-work-package, every commit/bug/validation. **Append here as work lands.** |
| `README.md` | User-facing description + limitations (what ships to someone who just wants to *run* cph). |
| `GPU_LAMBDA_UPDATE_PLAN.md` | The **current** GPU roadmap (variant B, GPU-resident λ-dynamics). L0–L3 done; L2/L5 remain. |
| `GPU_PORT_PLAN.md` | **Superseded** older "classic NB-offload" GPU sketch. Kept for reference; don't follow it. |
| `_portref/00_fork_vs_2021_all.diff` | The 2021 constant-pH fork's diff **vs pristine 2021.0**, per touched file (`00_stat.txt` = the ~46-file footprint). This is the source-of-truth for *what the fork changed*; the port re-implements each hunk adapted to 2026 APIs. Gitignored. |

## What this tree is

Pristine **GROMACS 2026.1** with the **constant-pH (λ-dynamics)** method of Aho,
Buslaev *et al.* (originally a GROMACS-2021 fork) ported onto it. The point is a
current-GROMACS cph build to replace the aging 2021 fork that runs the
`full_size_1d/r{1,2,3}_cph` umbrella-sampling campaign.

- **git baseline** = commit `b6b339e` ("pristine GROMACS 2026.1 source, pre-cph-port"). `git diff b6b339e HEAD` is the entire port (~77 src files, ~5400 insertions).
- **Branches:** `master` *is* the full CPU+GPU port (L0–L3 GPU-resident work plus the AA 1-4-pair fix). `gpu-lambda-update` was converged with `master` at `514bb20`; it has **not** followed since, so check before doing GPU work on it. Push only when asked.

### Other trees this one depends on / relates to (NOT part of this repo)

- **Fork oracle (2021, built):** `/home/semen/install/constantph/gromacs-constantph/build/bin/gmx` — the reference implementation every numerical validation compares against (single precision, thread-MPI, gcc-14). Confirmed present.
- **aurum2 CUDA build tree:** `/home/yesylevskyy/install/gromacs-2026-cph-gpu` (`build-cuda`) — used to build/test CUDA (there is no local GPU). See [[reference_aurum2_cuda_build]] and "Building & running on aurum2 (GPU)" below. **Since 2026-08-06 it is a git clone of `origin` tracking `master`, no longer an rsync** — update it exactly like the CPU tree below. The old rsync workflow is retired *because* its `--exclude 'build-*'` silently dropped 5 tracked files (`admin/ci-scripts/build-and-test-*.sh`, `build-taf-template.sh`, `docs/dev-manual/build-system.rst`) on top of the already-known `src/external/build-fftw/` and `src/buildinfo.h.cmakein`. Cluster-local artifacts (`build-cuda/`, `*_run/`, wrappers, logs) are listed in `.git/info/exclude`, so `git status` is clean.
- **aurum2 CPU build tree:** `/home/yesylevskyy/install/gromacs-2026-cph` (`build-sys`, plus `build_sys.sh`/`build_cuda.sh` wrappers that `spack load` and touch `BUILD_DONE`/`BUILD_FAIL`). Also a git clone of `origin`.
- **Updating either cluster tree** (once the work is pushed): `git fetch origin master && git merge --ff-only FETCH_HEAD`, then run that tree's build wrapper. If nothing has been pushed yet, use a **git bundle** instead (`git bundle create x.bundle <theirHEAD>..master`, scp, `git fetch /path/x.bundle master`) so you don't have to push unasked. ‼️ **Their `origin/master` ref goes stale, so `git rev-list HEAD..origin/master` can report "0 behind" while the tree is actually behind — always `git fetch` first.** The CPU tree had silently drifted **29 commits** once (pre-L0.1, no CPU PME reciprocal potential) and **3 commits** again on 2026-08-06; a bug report produced with its stale binary cost an extra debugging round. **Check `git log -1` there before trusting any result.**
- **Repro scripts + cph inputs** live in the *project* dir, not here: `/home/semen/work/Projects/Misha/CG/full_size/cph/` (`m0a_repro.sh`, `m0b_repro.sh`, `cph.mdp.inc`, `molecule_0.itp`, index/toppar, etc.). The M1 single-point validation inputs are assembled from `full_size/` + `full_size/cph/`.

## Repository layout — where the cph code lives

The method's own module (new, self-contained):

```
src/gromacs/applied_forces/constant_ph/
  constant_ph.{cpp,h}            # ConstantPH: the λ-dynamics driver (ODE integrator, groups, thermostat)
  read_params.{cpp,h}            # mdp/tpr parameter parsing
  update_topology_charges.{cpp,h}# baking λ→charges into the topology at grompp
  tests/constant_ph.cpp
src/gromacs/mdtypes/lambda_dynamics_params.h   # the parameter struct carried through inputrec/tpr
src/gromacs/tools/cphmd.{cpp,h}                # `gmx cphmd` analysis tool (reads the enxCPHMD edr block)
```

The port then threads the per-atom **electrostatic potential** (the dV/dλ driver) through
several stock subsystems — this is 90% of the difficulty. By subsystem:

- **Non-bonded kernels** (`src/gromacs/nbnxm/`): the kernels accumulate a per-atom potential buffer.
  - `kernels_reference/kernel_ref_{inner,outer}.h` + `kernel_ref_{1x1,4x4}.cpp` + `kernel_common.h` — plain-C reference path.
  - `simd_kernel.h` + `simd_kernel_inner.h` — the **SIMD** path (4xN / 2xNN); **this is the default for cph**.
  - `atomdata.{cpp,h}` — the `potential` output buffer + `reduceElectrostaticPotential` (nbat→atom) + the `computeElectrostaticPotential_` runtime flag.
  - `nbnxm.{cpp,h}`, `nbnxm_setup.cpp`, `kerneldispatch.cpp` — wiring, per-step buffer clear, kernel selection (cph forces plain-C 4x4 only under `GMX_CPH_NO_SIMD`).
  - `cuda/nbnxm_cuda_kernel.cuh`, `nbnxm_gpu_data_mgmt.cpp`, `gpu_types_common.h`, `cuda/nbnxm_gpu_buffer_ops_internal.cu` — the GPU NB potential path + device charge buffer.
- **PME reciprocal potential** (`src/gromacs/ewald/`): `pme.{cpp,h}`, `pme_gather.*` (`gatherPmePotential`), `pme_redistribute.cpp` (`dd_pmeredist_potential`), `pme_only.cpp`, GPU: `pme_gpu_*`, `pme_gather.cu`. **The reciprocal term is required for any PME system; RF-only Martini doesn't hit it.**
- **MD loop / force plumbing** (`src/gromacs/mdlib/`, `mdrun/`): `sim_util.cpp` (per-step charge re-push + reduce), `md.cpp` (buffer clear order, DD halo move of the potential, GPU-resident wiring), `forcerec.cpp` (`bHaveQ |= lambda_dynamics`), `energyoutput.cpp` (enxCPHMD block), `mdoutf.cpp`, `force.cpp`, `runner.cpp` (ConstantPH construction + `setCommrec`), `simulatorbuilder.*`, `domdec/mdsetup.cpp` (`updateAfterPartition`).
- **I/O** (`src/gromacs/fileio/`): `checkpoint.cpp` (λ x,v persist across `-cpi`), `tpxio.cpp` (`tpxv_ConstantpH`), `enxio.h`.
- **grompp** (`src/gromacs/gmxpreprocess/`): `readir.cpp` (mdp hooks), `grompp.cpp` (charge baking).
- **Task assignment** (`src/gromacs/taskassignment/`): `decidegpuusage.cpp`, `decidesimulationworkload.cpp` — the guards that decide when cph may use GPU / buffer-ops / GPU-update.

## Building — local CPU (`build-cpu`)

Configured (see `build-cpu/CMakeCache.txt`): `GMX_GPU=OFF GMX_MPI=OFF GMX_DOUBLE=OFF
GMX_FFT_LIBRARY=fftw3 BUILD_TESTING=OFF CMAKE_BUILD_TYPE=Release`, system gcc
(`/usr/bin/g++`), `GMX_SIMD=AUTO`.

```bash
cmake --build build-cpu -j16 --target gmx -- -k     # -k = make keep-going (cmake has no --keep-going)
# single object (fast iterate):
make -f build-cpu/src/gromacs/CMakeFiles/libgromacs.dir/build.make \
     build-cpu/src/gromacs/CMakeFiles/libgromacs.dir/<path>.cpp.o
```

### ‼️ THE RUNTIME GOTCHA (has cost hours, twice)

`build-cpu/bin/gmx` **silently loads the installed stock** `libgromacs`
(`/home/semen/programs/gromacs-2026.1/lib64/libgromacs.so.11`, put on
`LD_LIBRARY_PATH` by the user's GMXRC) instead of the rebuilt
`build-cpu/lib/libgromacs.so.11`. When that happens cph is **inert** — grompp reports
`Unknown left-hand 'lambda-dynamics'` and no hooks fire. `ldd build-cpu/bin/gmx | grep
libgromacs` right now points at the stock lib — so this is not hypothetical. **ALWAYS
run the port with the build lib prepended:**

```bash
LD_LIBRARY_PATH=/home/semen/install/gromacs-2026-cph/build-cpu/lib:$LD_LIBRARY_PATH build-cpu/bin/gmx ...
```

Verify: `ldd build-cpu/bin/gmx | grep libgromacs` must resolve into `build-cpu/lib`.

## Building & running on aurum2 (GPU)

There is **no local GPU** — all CUDA build/test happens on aurum2. Full recipe in
[[reference_aurum2_cuda_build]]; essentials:

- Tree: `/home/yesylevskyy/install/gromacs-2026-cph-gpu` — a **git clone** on `master` (see above). Wrappers `configure_cuda.sh` / `build_cuda.sh` live there (cluster-local, untracked) and encode everything below.
- Toolchain: `spack load cmake@3.31.11 cuda@13.1.1` (host gcc = system 11.5). **No `spack load fftw`** — see below.
- Configure: `cmake -S . -B build-cuda -DGMX_GPU=CUDA -DGMX_MPI=OFF -DGMX_DOUBLE=OFF -DGMX_FFT_LIBRARY=fftw3 -DGMX_BUILD_OWN_FFTW=ON -DBUILD_TESTING=OFF -DGMX_CUDA_TARGET_SM=86 -DCMAKE_BUILD_TYPE=Release -DGMX_SIMD=AVX2_256`.
- **‼️ `-DGMX_SIMD=AVX2_256` is mandatory:** login node is zen4 (AVX-512) but the GPU b-nodes are Zen3 (AVX2 only); a default build SIGILLs (mdrun exit 132) on the GPU nodes.
- **‼️ CPU-PME on GPU nodes needs `GMX_BUILD_OWN_FFTW=ON`, NOT `fftpack`.** The shared spack `fftw@3.3.10` is built for `linux-rocky9-zen4` (`-march=zen4`) and SIGILLs on Zen3 — that is the *only* reason fftpack was ever used, and **fftpack is very slow, so do not go back to it**. The GROMACS-managed FFTW configures `--enable-sse2 --enable-avx --enable-avx2` and relies on FFTW's *runtime* SIMD dispatch (no `-march`), so one static build is both fast and portable across zen4/Zen3. It downloads `fftw-3.3.10.tar.gz` (the login node has outbound HTTP); for an offline build point `GMX_BUILD_OWN_FFTW_URL` at a local tarball. Verified on a Zen3 b-node: `CPU FFT library: fftw-3.3.10-sse2-avx-avx2-avx2_128`, no SIGILL. Measured (2000 steps, 90k AA PME, in-place lib swap so the only variable is the FFT): PME 3D-FFT **2.95 s → 1.83 s** (`-nb cpu -pme cpu`) and **2.66 s → 1.55 s** (`-nb gpu -pme cpu -update gpu`), the latter **28.0 → 31.9 ns/day end-to-end (+14 %)**. No effect with `-pme gpu` (cuFFT; the CPU FFT is unused).
- **‼️ `gmx` is linked with DT_`RPATH`, which outranks `LD_LIBRARY_PATH`.** So copying a `gmx`+`libgromacs` pair aside and pointing `LD_LIBRARY_PATH` at it does **not** work — it silently loads whatever `libgromacs` is in `build-cuda/lib` *now*, and an A/B of two builds compares one build against itself. To benchmark two builds, swap `build-cuda/lib/libgromacs.so.11.0.0` in place (with a `trap` to restore it). Note this is the *opposite* direction of the local `build-cpu` gotcha above, where the stock lib wins.
- Compile on the login node (nvcc, no GPU needed); run/test via SLURM `--gres=gpu:1` on partition `b32_128_gpu` (sm_86 = RTX 3080/3090, max 4h, `--account=uochb --qos=normal`).
- Same `LD_LIBRARY_PATH=.../build-cuda/lib` prepend applies at runtime (plus `spack load cuda fftw`).

## Running cph — the run-level constraints (these are real, documented in README)

1. **`nstenergy > 0`, aligned to `lambda-dynamics-update-nst`.** The λ block only lands on energy-file frames. `nstenergy = 0` writes `nre=0` cph-only frames that `gmx cphmd`/`gmx energy` **cannot read** ("Energy header magic number mismatch"). Set `nstenergy = lambda-dynamics-update-nst`. This is a **config fix, not a code bug** — do NOT re-add the fork's `|| bCPHMD` frame-forcing (it manufactures exactly the unreadable frames). [[feedback_no_deffnm]] and the production mdps set this.
2. **CPU-NB requires `-nb cpu`** unless you're on a GPU build exercising the GPU path. On a CPU-only build a `lambda_dynamics` + GPU-NB run **fails fast** at setup (the guard in `nbnxm_setup.cpp` / `decidegpuusage.cpp`) — it used to silently give dV/dλ = 0.
3. **Single rank per simulation on the campaign build.** Multi-rank DD cph *is* implemented and validated (L0.2/L0.2b) but the campaign runs one rank per window; separate PME ranks (`-npme>0`) are guarded off for cph (run PME on the PP ranks).
4. **Leap-frog (`md`) integrator only**; λ thermostat: **no Langevin** (asserts). GPU-resident update needs **v-rescale/no** (not Nose-Hoover).
5. **`.tpr` is not portable** between the 2021 fork and this port (tpx version gap). Always `grompp` and `mdrun` with the same build. This is *why* every oracle comparison has each side grompp its own tpr.

### Debug/instrumentation env vars

| Var | Effect |
|---|---|
| `GMX_CPH_NO_SIMD=1` | fall back to plain-C 4x4 reference NB kernel (SIMD is the cph default) |
| `GMX_CPH_DUMP_DVDL=1` | write per-coord `dvdl_pot` to `cph_port_dvdl.dat` (the M1/M0a comparison artifact) |
| `GMX_CPH_DUMP_LAMBDAS=1` | write per-step all-group λ to `cph_port_lambdas.dat` |
| `GMX_NBNXN_PLAINC_1X1=1` | (stock) force the 1x1 reference kernel — used to pin the M1 reference path |

## Validation — oracles & how to compare against the fork

The port is validated **numerically against the 2021 fork**, not by unit tests. The
guiding principle for every gate: fixed RNG seeds (`lambda-dynamics-random-seed=12345`,
`-random-vv-seed=67890`) ⇒ both codes drive the *same* λ-integrator RNG ⇒ any dV/dλ
difference is purely the ported force path. Each side grompps its own tpr (tpx gap).

| Gate | System / mode | Target | Repro |
|---|---|---|---|
| **M1** | 46-λ RF Martini, single point (`nsteps 0`), reference kernel | all 46 groups dV/dλ vs fork **≤1e-4** (achieved 5e-5) | `run_m1.sh` (built per-session in scratch from `full_size/{,cph/}` inputs) |
| **M1 SIMD** | same, SIMD 4xN & 2xNN | vs fork 1e-4; vs reference kernel 6e-5 | `run_m1_simd.sh` |
| **M2** | multi-step, `nstlist=1` | stable, no potential-buffer accumulation; λ tracks reference ~1e-5 | — |
| **M0a** | 46-λ **PME** single point (`coulombtype=PME`) | all 46 vs fork, single-prec floor (rel 1.1e-5) | `full_size/cph/m0a_repro.sh` |
| **M0b** | RF **DD** (1 vs 2/4/8 ranks, 4×2 threads) | rel 3e-6; 30-step nstlist=10 matches single-rank | `full_size/cph/m0b_repro.sh` (also tests DD+PME) |
| **M1-GPU** | RF, `-nb gpu` (RTX 3080) single point | vs CPU oracle max rel 2.3e-6 | `m1gpu_run/job.sh` in the aurum2 `-gpu` tree |
| GPU PME / all-GPU / resident | 88k all-atom CHARMM36 PME, `-nb gpu -pme gpu [-update gpu]` | single-prec cuFFT floor (max ~6e-5); 1000-step λ = FP Lyapunov divergence only | `pmegpu_run/`, `bigpme_run/`, `traj1k/`, R0/R0b runs on aurum2 |
| **M6** | **90k all-atom CHARMM36m PME**, 54 λ coords, charge+multistate constraints — the first AA dV/dλ comparison **vs the fork** | single point worst rel **2.1e-5** (PME) / 5.0e-6 (RF) / **2.7e-6** at a displaced geometry with all λ off-centre; 60-step λ trace to 5.5e-5 (fork xvg precision) | inputs in `aurum2:~/work/Misha/CG/full_size_1d/aa_cph/pure_w113_fixlam/` (see `CPH_AA_BUG_REPRODUCER.md`) |

- **‼️ A single-point (`nsteps 0`) run cannot test any GPU gate.** `nsteps 0` is a **virial step**, and
  `useGpuFBufferOps = ... && !computeVirial` ⇒ on virial/energy steps buffer ops go **off**, the PME
  forces come back to the host, and the GPU-resident path silently degrades to the classic one. Two
  stale-buffer bugs (2026-08-06: PME reciprocal potential never D2H'd on the resident path; PME device
  charges never refreshed on the classic path) survived every gate for that reason. **Any GPU gate must
  run multiple steps with `nstcalcenergy` > run length**, so the ordinary-step schedule is what's tested.
- **‼️ Use an all-atom system for any new force-path gate.** M6 exists because every earlier gate was Martini (no `[ pairs ]`) or port-vs-port (GPU vs CPU, so a term missing from both cancels) — and that combination hid a completely absent 1-4 pair contribution for the whole port. Compare **against the fork**, on a topology **with 1-4 pairs**.
- **‼️ λ-trajectory comparisons vs the fork must run with `pcoupl = no` and `tcoupl = no`.** The 2021 and 2026 c-rescale / verlet-buffer implementations differ, so with coupling on the *atomic* trajectories separate within tens of steps and λ follows (0.024 by step 60 on M6 — not a cph bug; pinning `ld-seed` does not help). With coupling off, λ matched the fork to output precision over 60 steps.
- **Reference/oracle artifacts** (`M1_oracle_dvdl.txt` etc.) were built in session scratchpads and are ephemeral; the *inputs* to rebuild them are durable in `full_size/cph/`. When picking up validation, regenerate the oracle from the fork `gmx` + the repro script rather than trusting a stale file.
- Larger general-purpose PME validation system on the cluster: `/home/yesylevskyy/work/Misha/Rotate/pull_ang_ph` (88k-atom CHARMM36, 54 λ groups, charge+multistate constraints; use `w_3_selected/w_0/init.pdb` — `init.gro`/`phneutral.pdb` are unusable for single-point).

## Current status (as of tip `b8d8193`)

**CPU port: production-complete and validated** — grompp, multi-step MD (M1 5e-5, M2
stable), correct dV/dλ, SIMD kernel (~7.7× vs plain-C), edr + `gmx cphmd`, `-cpi`
checkpoint restart, DD (L0.2) and DD+PME (L0.2b). **All-atom force fields work as of
`2a9e5c1`** — before that the 1-4 pair (`[ pairs ]`) contribution to the per-atom
potential was missing entirely (M6; see `CPH_AA_BUG_REPRODUCER.md` and `PORT_LOG.md`).
Martini was unaffected because it has no `[ pairs ]`, which is why every earlier gate
passed.

**GPU port: L0–L3 done + validated** on RTX 3080 — GPU NB potential (L1.1), CUDA PME
potential (L1.2), and **GPU-resident** `-nb gpu -pme gpu -update gpu` (L3, host-λ /
R0+R0b). **Not done:** L2 (device group-reduce L2.2 + device charge-scatter L2.3 to
make per-step transfers O(Ngroups) instead of O(Natoms)); L5 (SYCL/HIP). **L2.2 was
attempted and dropped** — see the lesson below.

**Science validation (M3–M5) not done:** titration/pKa vs fork, a full production
window, refdata regression. That's science, not port work.

## Development conventions & gotchas

- **Understand the data-flow, then integrate cleanly — do not trial-and-error / monkey-patch.** [[feedback_get_the_logic]] This is the single most important rule for this codebase. The DD potential halo-exchange bug (L0.2) and the dropped L2.2 device reduce both came from placing a correct-looking operation at the *wrong point in the real pipeline*. Concretely:
  - **New quantities ride existing machinery at the same point.** The per-atom potential is communicated under DD by riding `dd_move_f` (the force halo move), reduced in the exact `reduceForces(NonLocal) → dd_move_f → reduceForces(Local)` order — because the home slots are forwarding scratch during the move, order is load-bearing (a `reduce(All)+move` corrupts multi-pulse forwarding). The PME reciprocal potential rides `dd_pmeredist_f`'s structure (`dd_pmeredist_potential`). Don't re-solve the comm separately.
  - **L2.2 dead-end (documented so it isn't repeated):** a device group-reduce bolted on *after* `do_force` in md.cpp reads potential buffers that `gpu_clear_outputs` has already zeroed / that lag on the PME stream. The fix for a future attempt is to run the reduce **inside `do_force`** at the potential-ready point (right after `pmeGpuWaitAndReduce` + the NB wait, ~`sim_util.cpp:2558`), before the buffers are cleared/reused. The WIP commits were `git reset --hard`'d away.
- **The port re-implements the fork adapted to 2026 APIs.** Use `_portref/00_fork_vs_2021_all.diff` as the spec for *what* to change; the *how* differs because of API churn. Key renames already applied: `gmx::index→Index`, `BOLTZ→c_boltz`, `emntUpdate→ModuleMultiThread::Update`, `warninp_t→WarningHandler*`, `search_string→find_group`, `t_commrec→MpiComm`, `IndexGroup` (do_index), moved headers (`vec`/`vectypes`→`utility`, `iserializer`→`serialization`), `InteractionFunction::{Potential,Kinetic}Energy`.
- **Silent-zero is the characteristic cph failure mode.** Reaching M1 uncovered two: a `#define` (`GMX_COMPUTE_ELECTROSTATIC_POTENTIAL`) not visible in a kernel TU compiled the accumulation out with no error; `updateAfterPartition` never being called left group atom-index lists empty. When dV/dλ comes back 0 or unchanged, suspect a compiled-out path or an unwired call, not a physics error.
- **The `bDoSplines |= !potentials.empty()` fix (PME).** `make_bsplines` skips charge-0 atoms; a titratable atom neutral at the current λ (charge 0 but Δq≠0, e.g. HIS at λ=1, BUF at λ=0.5) then gets no valid spline → garbage reciprocal potential. Force all-atom b-splines when gathering the potential. (No GPU analogue — `c_skipNeutralAtoms==false` there.)
- **`ConstantPH` is constructed early with a `nullptr` commrec** in `runner.cpp` (so a checkpoint can populate λ before `cr` exists); `setCommrec()` injects the real one later for DD. Single-rank use is unaffected.
- **Never use `mdrun -deffnm`** in cph setups — it renames `pullx.xvg`→`md.xvg` and is error-prone. [[feedback_no_deffnm]]
- **Keep `PORT_LOG.md` current.** It is the authoritative history (commit → WP → bug → validation). New work-package or gate → append a section; new limitation → also update `README.md`.

## Skeleton of the per-step cph data dependency (host, today)

```
setLambdaCharges(mdatoms.chargeA)      # λ → per-atom charges           [O(Natoms) write]
do_force  → NB kernel + PME fill V_i   # per-atom electrostatic potential (real + reciprocal)
reduceElectrostaticPotential → host    # V_i (nbat order) → fr.electrostaticPotential  [O(Natoms)]
updateLambdas: per group Σ φ·Δq, integrate the λ ODE (v-rescale thermostat, charge/multistate constraints)
```

The GPU-resident goal (variant B) keeps the two O(Natoms) ops (potential→group reduce;
charge scatter) on-device and leaves only the tiny O(Ngroups) λ ODE on the host. See
`GPU_LAMBDA_UPDATE_PLAN.md` §0 for the full dependency and layer plan.
