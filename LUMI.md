# LUMI.md — working on LUMI (access, EasyBuild, GROMACS-cph)

Practical guide for using **LUMI** (CSC) from this repo: how to log in, how to set up the
**EasyBuild** environment for our project, how to **use** the installed `GROMACS` cph module, and
how to **(re)compile** it. Most of this is generic to *any* software on LUMI — only the last
section is cph-specific. Companion to `CLAUDE.md` ("Building & running on LUMI") and
`HIP_PORT_PLAN.md`.

- **Login user:** `yesylevs`  ·  **Project (allocation):** `project_465003004`
- **GPU hardware:** LUMI-G node = 4× AMD **MI250X**, each = **2 GCDs** (so 8 GPU devices/node),
  arch **`gfx90a`**, wavefront **64**; host CPU = 1× AMD EPYC 7A53 "Trento" (Zen3).
- **Toolchain we standardize on:** software stack **`LUMI/25.09`**, **ROCm 6.4.4**,
  `cpeAMD/25.09`, EasyBuild **5.2.0**.

---

## 1. Access

```bash
ssh -i ~/.ssh/id_rsa_lumi yesylevs@lumi.csc.fi
```

You land on a login/UAN node (`uanNN`). Login nodes are for editing, compiling and submitting
jobs — **not** for running MD (use SLURM for that; see §6).

**Filesystems** (all Lustre; check with `lumi-quota`):

| Path | Purpose | Notes |
|---|---|---|
| `/users/yesylevs` (`$HOME`) | personal, small (22 G) | dotfiles, scripts |
| `/projappl/project_465003004` | project software / persistent | EasyBuild installs live here |
| `/scratch/project_465003004` | large, purged periodically | run directories, trajectories |
| `/flash/project_465003004` | fast NVMe scratch | hot I/O |

The login node has **outbound internet** (e.g. `git clone` from GitHub, EasyBuild source downloads).

### ‼️ Storage — run production in scratch, not `$HOME`

`$HOME` is small (22 G) and fills fast; **all production simulations and their output must run in
the project scratch area** `/scratch/project_465003004` (50 T, expandable). See the
[LUMI storage docs](https://docs.lumi-supercomputer.eu/storage/). Retention rules to respect:
**no LUMI storage is backed up**, and scratch is subject to **automatic cleaning if it fills** — so
the master copy of inputs and final results belongs in `/projappl` or off-cluster, not only in
scratch.

To keep paths convenient without polluting home, **create a per-user workspace in scratch and
symlink it into `$HOME`** (already done for `yesylevs`; the pattern for any project/user):

```bash
mkdir -p /scratch/project_465003004/$USER && ln -s /scratch/project_465003004/$USER ~/scratch
mkdir -p /flash/project_465003004/$USER   && ln -s /flash/project_465003004/$USER   ~/flash   # optional; fast I/O, 3x billed
```

Then run everything under `~/scratch/<run>/` (physically on scratch). Layout convention:

| Lives in | What | Persistent? |
|---|---|---|
| `/projappl/project_465003004/EasyBuild` | installed software (§2) | yes (50 G) |
| `$HOME/install/gromacs-2026-cph` | the git **source** tree (small; source only) | yes (22 G home) |
| `~/scratch/<run>/` = `/scratch/…/$USER/<run>` | **all run dirs, .tpr, trajectories, checkpoints, logs** | scratch (50 T, purgeable, no backup) |
| `~/flash/<run>/` | only when hot I/O is worth the 3× cost | flash (2 T) |

**Sync-tool / workflow implications (important):**
- The build & transfer flow (`git archive HEAD`, git bundles — §6) carries **source only** from the
  home tree and never touches scratch. **Do not git-track or bundle run inputs/outputs.**
- Job/run scripts must `cd ~/scratch/<run>` (or write outputs there), **not** into the source tree
  in `$HOME`. (The small gate/test scripts in `lumi/` run in-tree by design; *production* run
  scripts target scratch.)
- Scratch is not backed up and may be purged: keep the master copy of inputs under `/projappl` or on
  your workstation, and copy final results off scratch when a campaign finishes.

---

## 2. EasyBuild environment (generic — for any software)

LUMI installs user/project software with **EasyBuild**, into a *project* prefix so the whole
project can `module load` it. One-time-per-shell setup:

```bash
export EBU_USER_PREFIX=/project/project_465003004/EasyBuild   # WHERE installs go (shared)
module load LUMI/25.09        # software stack version
module load partition/G       # target partition: G=GPU, C=CPU, L=login (pick to match the HW)
module load EasyBuild-user    # activates eb, pointed at $EBU_USER_PREFIX
```

- `EBU_USER_PREFIX` **must be set *before* `module load LUMI …`** (the stack reads it). We keep it
  in `~/.bashrc`. It currently points at `project_465003004`; if you work on another project,
  override it per-session rather than editing `.bashrc` for a one-off.
- Installs land in a **shared** prefix (`…/EasyBuild/SW/LUMI-25.09/G/…`), visible to every project
  member — that is intended. The matching modules appear automatically in `module avail` once
  `EBU_USER_PREFIX` is set (no need to keep `EasyBuild-user` loaded just to *use* software; load it
  only to *install*).
- Convenience: the repo's LUMI tree has `~/install/lumi_cph_env.sh` that does the three loads
  above — `source` it. (Despite the name it is a generic EB env; nothing cph-specific.)

Verify:
```bash
eb --version                       # -> EasyBuild 5.2.0 (framework 5.2.0)
eb --show-config | grep -i installpath   # -> /project/project_465003004/EasyBuild/SW/LUMI-25.09/G
```

### Generic EasyBuild workflow

LUMI ships thousands of ready recipes in `/appl/lumi/LUMI-EasyBuild-contrib/easybuild/easyconfigs/`.

```bash
eb --search <name>                 # find a recipe (e.g. GROMACS-2026)
eb <recipe>.eb -D                  # dry run: parse + resolve deps, build nothing
eb <recipe>.eb -r                  # install, auto-resolving/building missing deps (--robot)
eb <recipe>.eb --rebuild           # force rebuild even if the module already exists
eb <recipe>.eb --ignore-checksums  # skip source checksums (needed for our custom/local sources)
clear-eb                           # wipe the build workdir ($EBU_USER_PREFIX build tmp)
```

Custom easyconfigs (ours) can live anywhere; pass the path. Local source tarballs go in
`$EBU_USER_PREFIX/sources/<first-letter>/<Name>/` (e.g. `.../sources/g/GROMACS/`).

---

## 3. LUMI gotchas (bite everyone — read once)

- **‼️ Never pipe `module load` / `source env.sh` through a pipe** (`| tail`, `| grep …`) in an ssh
  one-liner. A pipe runs the command in a **subshell**, so the PATH/module changes are lost in the
  parent shell — it looks like "the module didn't set PATH" but it did. Source/`module load`
  first, *then* run your command.
- **‼️ `LUMI/25.09` `LD_LIBRARY_PATH` warning.** The 25.09 stack's programming environment is newer
  than the system default (25.03); the module prints a note about adapting `LD_LIBRARY_PATH`. If you
  hit odd library mismatches, `module load lumi-CrayPath` *after* all other modules (and reload it
  after any later module change). Our GROMACS build has not needed it, but keep it in mind.
- **‼️ Binaries are linked with `RPATH`, which outranks `LD_LIBRARY_PATH`.** You cannot A/B two
  builds by copying a binary aside and pointing `LD_LIBRARY_PATH` at a different `libgromacs` — it
  loads the RPATH one. To compare builds, swap the library in place (with a `trap` to restore).
- **‼️ ISA / SIGILL.** Login/UAN nodes are Zen4; GPU compute nodes are **Zen3**. A default
  auto-SIMD build tuned on the login node **SIGILLs** on the compute node (mdrun exit 132). Build
  with `-DGMX_SIMD=AVX2_256` (portable across Zen3/Zen4). Likewise, host FFTW must not be
  `-march`-locked to Zen4.
- **‼️ FFT choices for GPU builds:** use **VkFFT** for the GPU FFT (`GMX_GPU_FFT_LIBRARY=VKFFT`).
  **rocFFT did not work for us on LUMI — do not retry it.** For CPU-side FFTW, use
  `GMX_BUILD_OWN_FFTW=ON` (runtime SIMD dispatch, portable); the shared spack `fftw@3.3.10` is
  built `-march=zen4` and SIGILLs on Zen3. **Never fall back to fftpack (very slow).**

---

## 4. Running on GPU nodes (SLURM)

Partitions (`sinfo`): `dev-g` (≤3 h, fast to schedule — use for tests), `small-g` (≤3 d),
`standard-g` (≤2 d). All expose `gpu:mi250:8` per node.

**Interactive (preferred for tests — skips the batch queue):**
```bash
salloc -A project_465003004 -p dev-g -N1 -n1 -c7 --gpus-per-node=1 -t 00:20:00 \
       bash -l my_script.sh
```
- **1 GCD = `--gpus-per-node=1`** with `-c7` (a GCD is NUMA-paired with ~7 usable cores).
- `salloc … <cmd>` runs `<cmd>` in the allocation and releases it — good for automation.
- Inside the allocation, launch the program with `srun -n1 -c7 --gpus-per-node=1 <prog> …`.

**Batch:** an `sbatch` script with `#SBATCH --account=project_465003004 --partition=small-g
--gpus-per-node=1 --cpus-per-task=7 -N1 -n1 -t …`, then `module load …` + `srun <prog>`.

- GPU selector env var is **`ROCR_VISIBLE_DEVICES`** (the HIP analogue of `CUDA_VISIBLE_DEVICES`);
  SLURM sets it from `--gpus-per-node`. `gmx mdrun -gpu_id` also works.
- Sanity on a node: `rocm-smi --showproductname` (should list `AMD Instinct MI250X`, `gfx90a`).
- For real performance runs, use the LUMI CPU↔GCD binding recipe (each GCD has a NUMA-local L3);
  for correctness tests, binding is irrelevant.

---

## 5. Using the installed GROMACS-cph module

The constant-pH GROMACS is installed in the shared project prefix:

```bash
module load LUMI/25.09
module load partition/G
module load GROMACS/2026.1-cpeAMD-25.09-cph-VkFFT-rocm-hip
which gmx_mpi          # -> …/SW/LUMI-25.09/G/GROMACS/2026.1-cpeAMD-25.09-cph-VkFFT-rocm-hip/bin
gmx_mpi --version      # GROMACS 2026.1, GPU support: HIP, GPU FFT: VkFFT, SIMD AVX2_256
```

The binary is **`gmx_mpi`** (MPI build). Run it under `srun` in an allocation (§4). Example
single-GCD cph run:
```bash
srun -n1 -c7 --gpus-per-node=1 gmx_mpi mdrun -s run.tpr -nb gpu -pme gpu -update gpu -ntomp 7
```

**GPU support for cph (HIP):** the per-atom-potential kernels exist for CUDA and HIP.
- **Classic path:** `-nb gpu -pme cpu` (RF) or `-nb gpu -pme gpu` (PME), with `-update cpu`.
- **Fully GPU-resident:** `-nb gpu -pme gpu -update gpu` — needs a **single GPU** and a system whose
  constraints GPU-update supports (**h-bonds / SETTLE OK; triangle constraints — e.g. Martini
  polarizable water — are refused on every backend**, so CG-PW systems must use the classic path).

**cph run-level constraints** (see `README.md` / `CLAUDE.md`, not LUMI-specific):
`nstenergy > 0` aligned to `lambda-dynamics-update-nst`; leap-frog integrator; no Langevin; single
rank per simulation on the campaign build; `.tpr` is **not portable** between builds — always
`grompp` and `mdrun` with the same binary.

---

## 6. (Re)compiling GROMACS-cph on LUMI

Source lives in the LUMI git clone `~/install/gromacs-2026-cph` (branch tracks the port; HIP work is
on `hip-port`). Cluster-local, untracked helpers are under `lumi/`.

**One command** (re-archives the current `git HEAD` → EasyBuild → shared prefix, ~7 min):
```bash
bash -l ~/install/gromacs-2026-cph/lumi/build_hip.sh
```
It sources the EB env, runs `git archive HEAD` into
`$EBU_USER_PREFIX/sources/g/GROMACS/gromacs-2026.1.tar.gz`, then
`eb lumi/GROMACS-2026.1-cpeAMD-25.09-cph-VkFFT-rocm-hip.eb --ignore-checksums --rebuild`, and
touches `BUILD_DONE` / `BUILD_FAIL`.

**The recipe** (`lumi/GROMACS-2026.1-cpeAMD-25.09-cph-VkFFT-rocm-hip.eb`) is adapted from the LUMI
contrib recipe `GROMACS-2026.0-cpeAMD-25.09-VkFFT-rocm-hip.eb` (Rasmus Kronberg / CSC); the only
change is the source (our local git-archive tarball, no upstream URL/checksum). Key CMake flags:
```
GMX_GPU=HIP   GMX_HIP_TARGET_ARCH=gfx90a   GMX_GPU_FFT_LIBRARY=VKFFT
GMX_MPI=ON    GMX_OPENMP=ON                GMX_BUILD_OWN_FFTW=OFF (cray-fftw)
C/CXX = ${ROCM_PATH}/llvm/bin/clang(++)    BLAS/LAPACK = libsci_amd
toolchain = cpeAMD/25.09
```

**To rebuild from a source change:** edit on the LUMI clone and commit, or (the usual flow) develop
on your workstation and move commits over with a **git bundle** — do **not** push unless asked:
```bash
# on the workstation:
git bundle create x.bundle <lastLumiCommit>..hip-port
scp -i ~/.ssh/id_rsa_lumi x.bundle yesylevs@lumi.csc.fi:~/install/gromacs-2026-cph/
# on LUMI:
cd ~/install/gromacs-2026-cph
git fetch x.bundle hip-port && git merge --ff-only FETCH_HEAD
bash -l lumi/build_hip.sh
```
‼️ The clone's `origin/*` refs go stale — always `git fetch` / check `git log -1` before trusting
that the tree is up to date.

**Building any other GROMACS flavour** (CPU, different version): copy the matching LUMI contrib
recipe (`eb --search GROMACS-2026`), adjust `sources`, and `eb … -r`. The env prep (§2) and gotchas
(§3) are the same.

---

## 7. Quick reference — files on the LUMI tree

| Path (`~/install/…`) | What |
|---|---|
| `lumi_cph_env.sh` | source to set up the EB env (generic) |
| `gromacs-2026-cph/lumi/build_hip.sh` | re-archive HEAD → `eb --rebuild` |
| `gromacs-2026-cph/lumi/GROMACS-2026.1-…-hip.eb` | the cph easyconfig |
| `gromacs-2026-cph/lumi/run_g1.sh` / `run_g2.sh` / `run_g3_aa.sh` | validation gates (NB / PME / resident) |
| `gromacs-2026-cph/lumi/run_smoke2.sh` | non-cph GPU smoke/regression (water) |
| `gromacs-2026-cph/lumi/m1_inputs/`, `aa_cph/` | staged test systems (Martini, CHARMM36) |
