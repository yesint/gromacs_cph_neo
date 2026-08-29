# HIP port plan: constant-pH GPU code CUDA → HIP (AMD MI250X / LUMI)

**Executive summary.** The constant-pH port threads one new physical quantity — the per-atom
electrostatic potential `V_i` (the dV/dλ driver) — through GROMACS's GPU machinery, plus a device
λ-charge buffer for the GPU-resident path. An audit of the full port delta (`git diff b6b339e HEAD`)
shows that **~85% of the GPU-side cph code is backend-agnostic and already compiles into a HIP
build**: all buffer management, D2H/H2D staging, synchronization changes, MD-loop wiring, and PME
host plumbing live in shared `.cpp`/`.h` files that `GMX_GPU=HIP` builds compile as HIP sources
(`nbnxm_gpu_data_mgmt.cpp`, `pme_gpu_internal.cpp`, `pme_gpu.cpp`, `gpu_types_common.h`,
`gpu_common.h`, `md.cpp`, `sim_util.cpp`). Only **three device-kernel edits** are CUDA-only and need
HIP twins: (1) the per-pair potential accumulation in the NB kernel, (2) the reciprocal-potential
gather in the PME gather kernel, (3) the λ-charge repack in the X buffer-op kernel. GROMACS 2026.1
ships a first-class native HIP backend (independent NB kernels, hipified-parallel PME kernels, full
buffer-ops/GPU-update/direct-comm support), so each CUDA edit has an exact, small landing site.
The plan is four work packages (H0 build baseline → H1 NB potential → H2 PME gather → H3 resident
charge repack) plus a one-line guard flip (H4), validated the same way the CUDA port was: multi-step
runs with `nstcalcenergy` > run length, dV/dλ and λ-trace compared against the same-binary CPU path,
which is itself fork-validated (M1/M0a/M6). Estimated effort: **1.5–3 weeks**, dominated by LUMI
queue latency, not code volume.

> **Scope: this is `GPU_LAMBDA_UPDATE_PLAN.md` Layer 5** ("multi-backend replication — only the
> kernels are per-backend; all host plumbing is backend-shared"). That prediction holds exactly:
> the L1.1/L1.2/L3 host plumbing needs zero HIP work. SYCL and OpenCL replication remain out of
> scope here (see §10). Target hardware: AMD MI250X, `gfx90a`, CDNA2, **wavefront 64**, on LUMI-G.

---

## 1. What has to exist on HIP (recap of the data flow)

Two GPU paths exist today on CUDA, both must work on HIP:

**Classic path** (`-nb gpu [-pme gpu]`, `-update cpu` ⇒ buffer ops forced OFF by
`decidesimulationworkload.cpp:174`):
```
host: setLambdaCharges → mdatoms->chargeA
      nbv->setAtomCharges (host nbat refresh)  +  gmx_pme_reinit_charges_gpu (PME d_coefficients H2D)
GPU:  full x+q H2D → NB kernel accumulates V_i into atdat.potential (nbat order)
      PME spread/solve/FFT/gather; gather also writes d_potentials (atom order)
host: potential D2H (gpu_launch_cpyback) → reduceElectrostaticPotential → + PME potentials_
      → host RF/Ewald self-term → updateLambdas (λ ODE)
```

**GPU-resident path** (`-nb gpu -pme gpu -update gpu`, buffer ops ON, single GPU only):
same, except x/v/f never leave the device; the X buffer-op kernel is the only per-step writer of
`xq`, so it re-packs `xq.w` from the device `lambdaCharges` buffer (uploaded per step,
`nbnxmGpuUploadLambdaCharges`); the potential D2H fires independently of the force buffer-op flag.

Everything in the "host:" lines is backend-agnostic. The three "GPU:" bold-faced operations —
NB `V_i` accumulation, PME gather `d_potentials` write, buffer-op `xq.w` repack — are the only
device code, and the only CUDA-specific code.

---

## 2. Exact delta to port (from `git diff b6b339e HEAD --stat`, GPU-relevant files)

Full port = 91 files / +7220 lines; most is CPU/method code. The GPU-relevant subset, classified:

### 2a. CUDA-only sources — **need a HIP twin** (the actual porting work)

| CUDA file (cph change) | What the port added | HIP landing site |
|---|---|---|
| `src/gromacs/nbnxm/cuda/nbnxm_cuda_kernel.cuh` (+32, ~line 676) | Per-pair potential accumulation, runtime-gated on `atdat.computePotential`: computes `coulFunc` per elec flavour (`EL_CUTOFF`: `int_bit*inv_r - c_rf`; `EL_RF`: `int_bit*inv_r + 0.5f*two_k_rf*r2 - c_rf`; `EL_EWALD_ANY`: `inv_r*(int_bit - erff(r2*inv_r*ewald_beta)) - int_bit*sh_ewald`), then `atomicAdd(&potential[ci*c_clusterSize+tidxi], epsfac*qj_f*coulFunc)` and `atomicAdd(&potential[aj], qi*coulFunc)` (qi pre-scaled by epsfac). RF/Ewald self-term deliberately host-side. Correctness-first per-pair atomics. | `src/gromacs/nbnxm/hip/nbnxm_hip_kernel_body.h` (§3, §7 H1) |
| `src/gromacs/ewald/pme_gather.cu` (+27) | `sumForceComponents` (line 247) gains `float* potential` out-param: `*potential += tdx.x*tdy.x*fxy1` (all-theta, no-derivative product). After `reduce_atom_forces` (~line 643): `__shfl_down_sync(c_fullWarpMask, potential, delta, atomDataSize)` loop delta=1,2,…, lane `splineIndex==0` writes `kernelParams.atoms.d_potentials[atomIndexGlobal]`. Gated `if constexpr (order==4 && numGrids==1)`. Kernel always writes (buffer sized `nAtomsAlloc`); the *copy-back* is what is gated on settings. | `src/gromacs/ewald/pme_gather_hip.cpp` (§5, §7 H2) |
| `src/gromacs/nbnxm/cuda/nbnxm_gpu_buffer_ops_internal.cu` (+32) | `nbnxn_gpu_x_to_nbat_x_kernel` gains `const float* gm_charge` param; when non-null, refreshes `gm_xq[threadIndex+offset].w = gm_charge[atomIndex]`. `launchNbnxmKernelTransformXToXq` passes `d_charge = nb->atdat->computePotential ? nb->atdat->lambdaCharges : nullptr`. | `src/gromacs/nbnxm/hip/nbnxm_gpu_buffer_ops_internal_hip.cpp` (§7 H3) |

### 2b. Backend-shared GPU sources — **zero porting; compile as HIP sources today**

Verified: `src/gromacs/nbnxm/CMakeLists.txt:97-100` adds `nbnxm_gpu_data_mgmt.cpp` +
`nbnxm_gpu_buffer_ops.cpp` to `HIP_SOURCES` under `if(GMX_GPU_HIP)`; `src/gromacs/ewald/CMakeLists.txt`
does the same for `pme_gpu.cpp`, `pme_gpu_internal.cpp` (lines ~112-142). The HIP NB module includes
the shared headers (`nbnxm_hip_types.h:51` includes `gpu_types_common.h` and uses the shared
`NBAtomDataGpu`; `nbnxm_hip.cpp:46` includes `gpu_common.h`; `pme_gather_hip.cpp:271` takes the shared
`PmeGpuKernelParams`). So all of the following cph changes are **already live in a HIP build**:

| Shared file (cph change) | What it does (needed on HIP, already there) |
|---|---|
| `nbnxm/gpu_types_common.h` (+13) | `NBAtomDataGpu::potential` (`DeviceBuffer<float>`), `computePotential` flag, `lambdaCharges` buffer — HIP kernels receive this struct **by value** (`nbnxm_hip_kernel_body.h:574`), so the new members already reach HIP device code. |
| `nbnxm/nbnxm_gpu_data_mgmt.cpp` (+78) | `gpu_init_atomdata` (:1004, :1053): alloc `potential` + `lambdaCharges` like `f`, sets `computePotential` from `nbat->computeElectrostaticPotential()`; `gpu_clear_outputs` (:1205): per-step clear; `gpu_launch_cpyback` (:1345): potential D2H into `nbatom->outputBuffer(0).potential`, **not** gated on force buffer-ops; `nbnxmGpuUploadLambdaCharges` (:1587): per-step H2D of λ charges on the local stream. All via `DeviceBuffer`/`copyTo/FromDeviceBuffer` — backend-generic. |
| `nbnxm/gpu_common.h` (+8, :307-321) | `gpu_try_finish_task`: `haveResultToWaitFor` extended with `\|\| nb->atdat->computePotential` so the resident path waits for the potential D2H. |
| `nbnxm/nbnxm_gpu.h` (+6), `nbnxm/nbnxm.cpp/.h` | `nbnxmGpuUploadLambdaCharges` decl; `nonbonded_verlet_t::uploadLambdaChargesToGpu` wrapper. |
| `ewald/pme_gpu_types.h` (+4) | `PmeGpuAtomParams::d_potentials` — inside the shared `PmeGpuKernelParams` (`pme_gpu_types.h:228`) that the **HIP** gather kernel already receives. |
| `ewald/pme_gpu_internal.cpp` (+48) | `pme_gpu_realloc_forces` (:254): realloc `d_potentials` + pinned `h_potentials` staging; `pme_gpu_copy_output_potentials` (:299): D2H (deliberately separate from the force copy, fires also under `useGpuForceReduction`); `pme_gpu_gather` (:2668): launches the potential D2H after the gather; `pme_gpu_getForceOutput` (:1183): surfaces `output->potentials_`; `pme_gpu_init` (:1420): `settings.computeElectrostaticPotential = pme->computePotential`. |
| `ewald/pme_gpu.cpp` (+28) | `pme_gpu_reduce_outputs` accumulates `output.potentials_` into the caller's `electrostaticPotential` ArrayRef; `pme_gpu_wait_finish_task` synchronizes when `computeElectrostaticPotential`; `pme_gpu_try_finish_task`/`wait_and_reduce` signatures extended. |
| `ewald/pme_gpu_settings.h`, `pme_gpu_staging.h`, `pme_gpu_types_host_impl.h`, `pme_output.h` | `computeElectrostaticPotential` flag; `h_potentials` pinned staging; `potentialsSize/SizeAlloc`; `PmeOutput::potentials_`. |
| `ewald/pme.cpp:783` / `pme.cpp:1882` / `pme.h:300` | `pme->computePotential = ir->lambda_dynamics`; `gmx_pme_reinit_charges_gpu()` — per-step H2D refresh of PME `d_coefficients` (plain re-copy at unchanged size, grid 0 only). |
| `mdlib/sim_util.cpp` (+44) | Per-step host-nbat charge re-push (`nbv->setAtomCharges`, :1793); passes `fr->electrostaticPotential` into `gmx_pme_do` / `pmeGpuWaitAndReduce` / `alternatePmeNbGpuWaitReduce` (:2247, :2296, :2506, :2517). |
| `mdrun/md.cpp` (+165) | Pre-force: `setLambdaCharges`, buffer zero, `uploadLambdaChargesToGpu` (resident), `gmx_pme_reinit_charges_gpu` (any GPU-PME) (:1251-1290). Post-force: NB potential reduce (+DD halo ride on `dd_move_f`), **host-side GPU self-term** `V_a -= facel·q_a·2Vc_sub_self` gated on `fr_->nbv->useGpu()` — backend-agnostic, applies to HIP automatically (:1321-1443); `updateLambdas`; dump hooks. |
| `listed_forces/listed_forces_gpu_impl.cpp:147` | cph forbids GPU bondeds (1-4 pairs contribute to `V_i` on the CPU only) — backend-agnostic, stays. |

**Consequence:** after a plain `GMX_GPU=HIP` build of this tree, a cph run would already allocate,
clear, copy back, and reduce `atdat.potential` and PME `d_potentials` — **as zeros**, because no HIP
kernel writes them. That is precisely the silent-dV/dλ=0 failure mode the port guards against
(README limitation 1), which is why `decidegpuusage.cpp:221` currently refuses GPU NB for cph on
`!GMX_GPU_CUDA` builds. The port work is exactly: make the three kernels write, then relax that guard.

### 2c. Guards added by the port (audit: the only `GMX_GPU_CUDA` in the whole diff is one line)

| Guard | Location | HIP action |
|---|---|---|
| GPU NB refused for cph on non-CUDA | `taskassignment/decidegpuusage.cpp:221` — `errorReasons.appendIf(ir.lambda_dynamics && !GMX_GPU_CUDA, ...)` in `canUseGpusForNonbonded` | **The one line to change** (H4): `!(GMX_GPU_CUDA \|\| GMX_GPU_HIP)`; reword the message. PME-on-GPU requires NB-on-GPU, so this single guard also gates the PME potential path — hence flip it only when **both** H1 and H2 are in the tree (§7). |
| Resident update needs single GPU | `decidegpuusage.cpp:796` (`lambda_dynamics && (isDomainDecomposition \|\| havePmeOnlyRank)`) | Keep as-is (device group-reduce / PME→PP potential comm are separate future WPs; identical on HIP). |
| Separate PME ranks fatal | `decidesimulationworkload.cpp:132` | Keep as-is. |
| Buffer-ops policy (`cphForcesBufferOpsOff = lambda_dynamics && !useGpuForUpdate`) | `decidesimulationworkload.cpp:174` | Keep as-is — backend-agnostic; gives classic path w/o `-update gpu`, resident path with it. |
| GPU emulation fatal | `nbnxm/nbnxm_setup.cpp:320` | Keep as-is. |
| GPU bondeds refused | `listed_forces_gpu_impl.cpp:147` | Keep as-is (HIP has GPU bondeds — `capabilities.h: Bonded` true — but the 1-4 potential term only exists in `listed_forces/pairs.cpp` on the CPU; offloading would re-open the M6 bug). |

---

## 3. The HIP backend in this tree (GROMACS 2026.1)

- **Selection:** `-DGMX_GPU=HIP` (`cmake/gmxManageHipccConfig.cmake`), compiled with
  `CMAKE_HIP_COMPILER=amdclang++`, min ROCm 5.2. Install guide: "In GROMACS 2026 there is full
  support for using HIP as the GPU backend on AMD devices" (`docs/install-guide/index.rst:1170ff`).
- **Capabilities** (`src/gromacs/gpu_utils/capabilities.h`): for HIP, `Nonbonded`, `Pme`,
  `BufferOps`, `Update`, `Bonded`, `StreamQuery`, thread-MPI and lib-MPI direct comm are all **true**
  (only `NonbondedFE` is CUDA-only — irrelevant to cph, which is not FEP). So both cph GPU paths
  (classic and GPU-resident) are architecturally available on HIP.
- **Wavefront:** for a pure-CDNA/GCN target list (e.g. `GMX_HIP_TARGET_ARCH=gfx90a`),
  `GMX_GPU_NB_DISABLE_CLUSTER_PAIR_SPLIT` defaults **ON** (`gmxManageHipccConfig.cmake:137-141`) →
  cluster-pair split 1 → NB parallel execution width = 8×8 = **64** (one wavefront covers a whole
  cluster pair; CUDA uses split 2 / width 32). PME kernels are templated on
  `parallelExecutionWidth` and instantiate both 32 and 64 (`pme_gather_hip.cpp:568-569`), selected at
  runtime from `deviceInfo.supportedSubGroupSizes[0]` (`pme_gpu_program_impl_hip.cpp:76-79`).
- **hipcc flags:** `-munsafe-fp-atomics` is enabled (`gmxManageHipccConfig.cmake:155`) →
  `atomicAdd(float*)` compiles to native `global_atomic_add_f32` on gfx90a (fast, not a CAS loop).

### Is the HIP NB kernel a hipified CUDA copy? — **No: independent implementation.**

`nbnxm/hip/nbnxm_hip_kernel_body.h` is a modern rewrite: one C++ template
`nbnxmKernel<doPruneNBL, doCalcEnergies, elecType, vdwType, nthreadZ, minBlocksPerMp, pairlistType>`
using `if constexpr` on `EnergyFunctionProperties` (vs CUDA's `#ifdef EL_RF` macro-multiplied
`.cuh`), AMD-specific fast paths (`AmdFastBuffer`, `AmdPackedFloat3`, `amdFastAtomicAddForce`,
`__builtin_amdgcn_readfirstlane`, DPP shuffles). It is explicitly instantiated in **four TUs**:
`nbnxm_hip_kernel_body_{f,fv}_{prune,noprune}.cpp` — so the cph edit goes in the **one header** and
covers every kernel flavour. The PME HIP kernels (`pme_gather_hip.cpp`, `pme_spread_hip.cpp`,
`pme_solve_hip.cpp`) are by contrast structurally parallel to the CUDA `.cu` files (same function
names, same shared-memory layout, extra `parallelExecutionWidth` template parameter, DPP used in the
stock force reduce) — porting the `pme_gather.cu` hunks there is nearly mechanical.

### CUDA→HIP file map (where each cph kernel change lands)

| CUDA file (cph change exists) | HIP counterpart (change must go) | Relationship |
|---|---|---|
| `nbnxm/cuda/nbnxm_cuda_kernel.cuh` | `nbnxm/hip/nbnxm_hip_kernel_body.h` (inner j-loop, lines ~989-1043) | independent implementations; re-derive, don't transliterate |
| `nbnxm/cuda/nbnxm_gpu_buffer_ops_internal.cu` | `nbnxm/hip/nbnxm_gpu_buffer_ops_internal_hip.cpp` (kernel :73, launch :112) | near-identical clone; mechanical |
| `ewald/pme_gather.cu` | `ewald/pme_gather_hip.cpp` (`sumForceComponents` :169, kernel :271, after `reduceAtomForces` call :444) | structurally parallel; mechanical with shuffle-API swap |
| `nbnxm/cuda/nbnxm_cuda_data_mgmt.cpp` (untouched by port) | `nbnxm/hip/nbnxm_hip_data_mgmt.cpp` (platform hooks only) | nothing to do — the cph data mgmt is in the shared file |
| `ewald/pme_gpu_program_impl.cu` (untouched by port) | `ewald/pme_gpu_program_impl_hip.cpp` | nothing to do — kernel selection unchanged (the gather kernel always writes `d_potentials`; the D2H is what is gated) |
| — (SYCL twins, out of scope) | `nbnxm/sycl/*`, `ewald/pme_gather_sycl.cpp` | future L5b |

---

## 4. API mapping for the primitives the cph kernels use

| Primitive in the cph CUDA code | HIP (gfx90a) equivalent | Already abstracted by GROMACS? |
|---|---|---|
| `atomicAdd(float*, float)` (per-pair V accumulation) | `atomicAdd` — native `global_atomic_add_f32` under `-munsafe-fp-atomics` (set globally) | Used directly in stock HIP kernels (`nbnxm_hip_kernel_body.h:421`, `pme_spread_hip.cpp`); no wrapper needed |
| `__shfl_down_sync(c_fullWarpMask, v, delta, width)` (PME potential reduce) | `__shfl_down(v, delta, width)` — **no `_sync` variants, no lane mask in HIP**; sub-width shuffles (width=`atomDataSize`≤16) behave identically | The exact pattern already exists in stock `pme_gather_hip.cpp:128-131` (`__shfl_down(fx, delta, width)`); copy that style, **do not port `c_fullWarpMask`** |
| `erff()` | `erff()` — same device libm name | stock HIP NB kernel already calls it (`:1032`) |
| Warp size **32** | Wavefront **64** on CDNA (RDNA is 32) — see per-kernel analysis below | NB: `sc_gpuParallelExecutionWidth(pairlistType)`; PME: `parallelExecutionWidth` template param |
| `float3/float4`, `asFloat3` | same via `gpu_utils/vectype_ops_hip.h`, `typecasts_cuda_hip.h` | yes |
| `DeviceBuffer<float>`, `allocateDeviceBuffer`, `clearDeviceBufferAsync`, `copyTo/FromDeviceBuffer`, `GpuApiCallBehavior::Async` | identical (`gpu_utils/devicebuffer_hip.h`) | yes — this is why all data mgmt is zero-port |
| `prepareGpuKernelArguments` / `launchGpuKernel` / `KernelLaunchConfig` | identical (`gpu_utils/hiputils.h`) | yes |
| Pinned host staging (`PaddedHostVector` + `changePinningPolicy`) | `gpu_utils/pmalloc_hip.cpp` | yes |
| `__launch_bounds__(t, b)` | supported by hipcc (stock HIP kernels use it) | direct |
| `__restrict__`, `if constexpr`, `#pragma unroll` | identical | direct |

### Warp-size risk analysis, per cph kernel piece (the headline risk — mostly defused)

1. **NB potential accumulation — warp-size independent.** The CUDA implementation is per-thread,
   per-pair `atomicAdd` with **no lane communication**: no shuffles, no ballots, no assumptions on
   which lanes share an i- or j-atom. The index arithmetic (`ai = ci*c_clSize + tidxi`,
   `aj = cj*c_clSize + tidxj`) is identical in the HIP kernel (`:701`, `:819`). Porting it 1:1 into
   the 64-wide HIP kernel is safe. (A later warp-level reduction optimization *would* be
   width-sensitive — deferred to H5.)
2. **Buffer-op charge repack — warp-size independent.** 1 thread : 1 atom, straight-line.
3. **PME gather potential reduce — the only lane-communication code.** It reduces each atom's
   `atomDataSize` partial sums (16 for `ThreadsPerAtom::OrderSquared`, 4 for `Order`) with
   `shfl_down` over sub-groups of `atomDataSize` lanes. On CDNA the wavefront holds 64/16 = 4 atoms
   (vs 2 on CUDA), but the invariant that matters — each atom's lanes are *contiguous and aligned*
   within the wavefront — is already load-bearing in the **stock** HIP force reduce
   (`pme_gather_hip.cpp:128`, `__shfl_down(fx, delta, width)` with `width = atomDataSize`), so the
   potential reduce piggybacks on a proven layout. Start the loop at `delta = 1` (the scalar has no
   3-component transposition, unlike the force reduce's DPP prologue). Risk: low, but this is the
   piece to eyeball in review and to bracket with the "every λ-atom's V_i non-zero" check.
4. **Charge-scaling convention (not warp, but the classic sign/factor trap).** Both kernels
   pre-scale the shared i-charge by `epsfac` (CUDA: `qi` shared scaled; HIP: `xqbuf.w *= epsFac` at
   `:711`) and keep the j-charge raw. So on HIP, exactly as on CUDA:
   `V_i += epsFac * qj * coulFunc` and `V_j += qi_scaled * coulFunc`. Getting this wrong is a clean
   ×~138.9 (epsfac) error the H1 gate would catch instantly — but know it going in.
5. **Exclusion mask type.** CUDA uses `int_bit` (float 0/1); HIP uses `pairExclMask` (float 0/1) —
   same semantics, and the HIP `energyElec` expressions are *already* the exact per-pair potential
   forms with `qi*qj` factored off (see §7 H1). The self-pair diagonal masking
   (`nonSelfInteraction`, `notExcluded`) matches the CUDA structure; the i==j self-term stays on the
   host (md.cpp block, already gated on `nbv->useGpu()` — fires for HIP unchanged).

---

## 5. PME on AMD: VkFFT is a non-issue for the potential gather

The LUMI/AMD default is `GMX_GPU_FFT_LIBRARY=VkFFT` (CMakeLists.txt:328-343; rocFFT optional). The
cph PME addition is **entirely inside the gather stage**: it reads `kernelParams.grid.d_realGrid[0]`
— the real-space potential grid *after* solve and inverse FFT — and forms Σ θx·θy·θz·grid per atom.
It touches no FFT plan, no library API, no complex grid, and the D2H/staging is shared host code.
**There is zero cuFFT/VkFFT/rocFFT coupling; the FFT library choice only moves the single-precision
numerics floor.** (The CUDA validation floor was max rel ~6e-5 on an 88k-atom system with cuFFT;
VkFFT's floor must be *measured*, not assumed, before fixing the H2 gate tolerance — see §9.)

HIP landing site: `ewald/pme_gather_hip.cpp` —
- `sumForceComponents` (`:169`): add the `float* potential` out-param and the
  `*potential += tdx.x * tdy.x * fxy1;` line (both `tdx`/`tdy` are (value, derivative) float2 pairs,
  identical to CUDA — `.x` is the value).
- `pmeGatherKernel` (`:271`): per-thread `float potential = 0.0F;`, pass `&potential` to both
  `sumForceComponents` calls (`:423`, `:480`; the numGrids==2 call's value is unused, as on CUDA).
- After the `reduceAtomForces` call (`:444`) + `__syncthreads()`: the `if constexpr (order == 4 &&
  numGrids == 1)` block with a `__shfl_down(potential, delta, atomDataSize)` loop (delta = 1,2,…<
  atomDataSize) and the `splineIndex == 0` store to `kernelParams.atoms.d_potentials[atomIndexGlobal]`.
- Instantiations (`:553-569`) unchanged — the potential write compiles into all
  order-4 variants for both widths 32 and 64.

`c_skipNeutralAtoms == false` on all GPU backends (`pme_gpu_constants.h`), so the L0.1 CPU
`bDoSplines` subtlety has no HIP analogue (same as CUDA — titratable atoms neutral at current λ
still get valid splines).

`pme_solve_hip.cpp`, `pme_spread_hip.cpp`, `pme_gpu_program_impl_hip.cpp`: **no changes** (spread
already uploads the coefficients that `gmx_pme_reinit_charges_gpu` refreshes; solve/energy is
untouched by cph on GPU).

---

## 6. Guard changes — the exact edits (all in H4)

1. `src/gromacs/taskassignment/decidegpuusage.cpp:221` (`canUseGpusForNonbonded`):
   change the condition `ir.lambda_dynamics && !GMX_GPU_CUDA` →
   `ir.lambda_dynamics && !(GMX_GPU_CUDA || GMX_GPU_HIP)` and update the message text ("implemented
   for CUDA and HIP only"). **This is the entire enablement switch** — because PME-on-GPU requires
   NB-on-GPU, it gates both kernels; because `-nb auto` consults it, non-ported backends (SYCL,
   OpenCL) keep falling back to CPU NB with a logged reason. Therefore it must flip **only in the
   same commit range that contains both H1 and H2** — flipping after H1 alone would let
   `-nb gpu -pme gpu` run with a zero reciprocal potential (allocated, copied back as zeros — the
   exact silent-zero failure the guard exists to prevent). During H1 bring-up on a dev branch,
   either test RF-only systems, or temporarily flip the guard *and* add a scratch fatal for
   `lambda_dynamics && useGpuPme` — but never merge that intermediate state to `master`.
2. `README.md` limitation 1: rewrite "GPU support is CUDA only" → CUDA + HIP; SYCL/OpenCL still
   guarded. `PORT_LOG.md`: new L5/HIP section.
3. Everything else (resident-single-GPU `decidegpuusage.cpp:796`, separate-PME fatal
   `decidesimulationworkload.cpp:132`, buffer-ops policy `:174`, emulation fatal
   `nbnxm_setup.cpp:320`, GPU-bondeds refusal `listed_forces_gpu_impl.cpp:147`): **unchanged** —
   all backend-agnostic and still correct on HIP.

---

## 7. Work packages (layered, each ends in a gate)

Ordering rationale: H0 de-risks the build+oracle before any code; H1 before H2 because RF systems
let NB validate without PME; H3 (resident) last because it depends on nothing new besides the
buffer-op kernel and is the least critical for first science use. Do the work on a branch
(suggest `hip-port` off `master`); merge to `master` only at H4 with the guard flipped once.

### H0 — LUMI build + oracle baseline (no cph code changes) — ~2-4 days incl. queue time
- Clone this repo on LUMI; configure a HIP build (see §8). This is the **first compile of the cph
  additions under hipcc** — the shared files (`nbnxm_gpu_data_mgmt.cpp`, `pme_gpu_internal.cpp`,
  `pme_gpu.cpp`, …) are compiled as HIP sources, so any hipcc-vs-nvcc C++ friction surfaces here.
  Expected clean (the additions are plain C++ over the abstraction layer), but this is the test.
- Also build a CPU-only config on LUMI (or use `-nb cpu` on the HIP build) and re-run one known CPU
  gate (M1 RF and/or M0a PME single point, inputs from `full_size/cph/`): dV/dλ vs the local
  `build-cpu` artifact to ~1e-6 rel (same code; different compiler/libm → tiny FP noise only).
  **This establishes the on-LUMI oracle**: the oracle chain is 2021 fork → local CPU port
  (M1 5e-5 / M0a 1.1e-5 / M6 2.1e-5) → LUMI CPU path → LUMI HIP GPU.
- Sanity: a stock (non-cph) HIP GPU run on one GCD (any benchmark system) to prove toolchain + node.
- **Gate G0:** LUMI CPU cph dV/dλ == local CPU port; stock HIP run clean. Risk: low (logistics).

### H1 — HIP NB per-atom potential — ~2-3 days
- **File:** `src/gromacs/nbnxm/hip/nbnxm_hip_kernel_body.h`, inner j-loop, immediately after the
  force accumulation (`fCiBuffer[i] -= forceIJ;`, line ~1042), inside the
  `(r2 < rCoulombSq) && notExcluded` branch. Runtime-gated on `atdat.computePotential` (hoist
  `float* potential = atdat.potential; const bool doPotential = atdat.computePotential;` next to the
  other `atdat` unpacks, ~line 600) — **outside** any `if constexpr (doCalcEnergies)`: the gates run
  F-only kernels (`nstcalcenergy` > run length), and those are exactly the kernels that must
  accumulate.
- Per-pair value: reuse the kernel's own energy expressions with `qi*qj` factored off — they are
  literally the CUDA `coulFunc` forms already present at `:1022-1033`:
  - `props.elecCutoff`: `pairExclMask * rInv - cRF`
  - `props.elecRF`: `pairExclMask * rInv + 0.5F * twoKRf * r2 - cRF`
  - `props.elecEwald` (covers **both** `elecEwaldAna` and `elecEwaldTab` — the potential is the
    plain erfc form regardless of how the *force* is tabulated, exactly as CUDA's single
    `EL_EWALD_ANY` expression): `rInv * (pairExclMask - erff(r2 * rInv * ewaldBeta)) - pairExclMask * ewaldShift`
  Then `atomicAdd(&potential[sci * c_clusterPerSuperCluster * c_clSize + i * c_clSize + tidxi],
  epsFac * qj * value)` and `atomicAdd(&potential[aj], qi * value)` (`qi` = `xqibuf.w`, already
  epsFac-scaled; `qj` raw — §4 item 4). Correctness-first per-pair atomics, exactly like the CUDA
  L1.1 commit (7bb98be); one edit covers all four kernel TUs and both 32/64-wide layouts.
- No launch/arg plumbing: `atdat` is passed by value and already carries the new members.
- **Gate G1 (M1-GPU-HIP):** 46-λ RF Martini (M1 inputs), 1 GCD, `-nb gpu -pme cpu -update cpu`
  (classic path; buffer ops auto-off), **multi-step** (≥20 steps, `nstcalcenergy` > nsteps,
  `pcoupl/tcoupl = no`), vs same-binary `-nb cpu`: step-0 dV/dλ (`GMX_CPH_DUMP_DVDL`) all 46 groups
  ≤ ~1e-5 rel (CUDA achieved 2.3e-6); λ trace (`GMX_CPH_DUMP_LAMBDAS`) tracks to FP noise; every
  λ-atom `V_i` non-zero (the silent-zero check). Repeat on the 90k AA system (M6 inputs) with
  `coulombtype = reaction-field` if feasible, to exercise the AA/1-4 interplay on-node.

### H2 — HIP PME reciprocal potential — ~1-2 days
- **File:** `src/gromacs/ewald/pme_gather_hip.cpp`; the four edits itemized in §5. Use
  `__shfl_down(potential, delta, atomDataSize)` (no mask), loop from `delta = 1`. All host plumbing
  (alloc, D2H, staging, reduce into `fr->electrostaticPotential`, `gmx_pme_reinit_charges_gpu`) is
  already live for HIP.
- **Gate G2:** AA PME system (M6 `pure_w113_fixlam` 90k inputs, or the cluster 88k `pull_ang_ph`
  system), 1 GCD, `-nb gpu -pme gpu -update cpu`, multi-step with `nstcalcenergy` > nsteps: dV/dλ vs
  same-binary CPU. Tolerance: start from the CUDA/cuFFT floor (max rel 6.2e-5 on 88k) but **first
  measure the VkFFT floor** on a non-cph observable (PME force/energy GPU-vs-CPU) and set the gate
  to that floor ×~3. Also re-run G1's RF case to confirm no regression.

### H3 — HIP resident path: X buffer-op charge repack — ~1-2 days
- **File:** `src/gromacs/nbnxm/hip/nbnxm_gpu_buffer_ops_internal_hip.cpp`. Mechanical mirror of the
  `.cu` change: `const float* __restrict__ gm_charge` kernel param (+ null check writing
  `gm_xq[threadIndex + offset].w = gm_charge[atomIndex]`), and in
  `launchNbnxmKernelTransformXToXq` (`:112`) the extra
  `d_charge = nb->atdat->computePotential ? ... lambdaCharges : nullptr` kernel argument.
  Everything else on the resident path (per-step `nbnxmGpuUploadLambdaCharges` H2D, PME coefficient
  refresh, potential D2H under buffer-ops, the `gpu_common.h` wait) is shared code, already active.
- **Gate G3 (R0-HIP / R0b-HIP):** the AA PME system, 1 GCD, `-nb gpu -pme gpu -update gpu`
  (`tcoupl = v-rescale` or `no`; `pcoupl = no` for the trace): (a) multi-step short run,
  `nstcalcenergy` > nsteps, dV/dλ at the first output vs CPU at the G2 floor; (b) 1000 steps,
  `nstlist = 20`: λ trace vs the classic-path (`-update cpu`) HIP run — identical through the early
  steps, then FP Lyapunov divergence only (CUDA reference: identical to step ~50, 2.5e-3 @ 1000).
  ‼️ **Never gate this on `nsteps 0`** — a single point is a virial step, buffer ops turn off, and
  the resident path silently degrades to classic (this hid two stale-buffer bugs on CUDA; see
  CLAUDE.md / PORT_LOG 2026-08-06). Multi-step with `nstcalcenergy` > run length is mandatory.

### H4 — guard flip + docs + regression — ~1 day
- The `decidegpuusage.cpp:221` edit (§6), README limitation 1, PORT_LOG entry.
- Regression battery on LUMI: G1 + G2 + G3 re-run on the final commit; one non-cph HIP run
  byte-compared / energy-compared against the pre-cph baseline (all cph buffers and branches are
  gated on `lambda_dynamics`, so stock behaviour must be untouched); CUDA build on aurum2 re-run
  (M1-GPU + R0b) to prove no cross-backend regression from the shared-file edits (there should be
  none — H1-H3 touch only `hip/` + `pme_gather_hip.cpp`, H4 touches one guard line).

### H5 (optional, later) — performance pass
- Profile (rocprof / omniperf) the per-pair `atomicAdd` cost on gfx90a. Both adds have ~8-way
  same-address conflicts per instruction across the wavefront (i-add: constant `tidxi` column;
  j-add: constant `tidxj` row). Options, in order of effort: register-accumulate the j-potential
  over the i-loop (mirror `fCjBuf`) → one atomic per (jm, thread); i-potential in a per-i register
  array (mirror `fCiBuffer`) reduced at the end alongside `reduceForceI`; LDS staging;
  `amdDppUpdateShfl`-based reductions. Also consider a fused device group-reduce — but that is the
  L2.2 territory with a documented dead-end: any such reduce must run **inside** `do_force` at the
  potential-ready point, not after it (see CLAUDE.md, "L2.2 dead-end").
- Only after G1-G3 pass and only if the potential shows up in profiles; correctness ships first.

---

## 8. LUMI build & run specifics

- **Hardware:** LUMI-G node = 4× MI250X; each card exposes **2 GCDs = 2 GPU devices** (8 per node),
  `gfx90a`, wavefront 64; host CPU = 1× AMD EPYC 7A53 "Trento" (Zen3, 64 cores). "Single GPU" in the
  resident-path guard means **one GCD**. The campaign pattern (one umbrella window per GCD,
  single rank) fits the validated single-GPU resident configuration exactly; multi-GCD DD runs fall
  back to the classic path automatically (guard `decidegpuusage.cpp:796`).
- **Configure** (mirror of the repo's aurum2 wrapper discipline — put it in a cluster-local
  `configure_hip.sh`/`build_hip.sh`, untracked):
  ```
  cmake -S . -B build-hip \
        -DCMAKE_HIP_COMPILER=${ROCM_PATH}/bin/amdclang++ \
        -DCMAKE_PREFIX_PATH=${ROCM_PATH} \
        -DGMX_GPU=HIP -DGMX_HIP_TARGET_ARCH=gfx90a \
        -DGMX_GPU_FFT_LIBRARY=VkFFT \
        -DGMX_MPI=OFF -DGMX_DOUBLE=OFF -DGMX_FFT_LIBRARY=fftw3 \
        -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DGMX_SIMD=AVX2_256
  ```
  Notes: `gfx90a`-only target ⇒ `GMX_GPU_NB_DISABLE_CLUSTER_PAIR_SPLIT` auto-ON (verify in the CMake
  cache) and `GMX_ENABLE_AMD_RDNA_SUPPORT` auto-OFF — both wanted. `GMX_SIMD=AVX2_256` is correct
  for Zen3 compute nodes and immunizes against login/compute ISA mismatch (the aurum2 SIGILL
  lesson). CPU FFTW: LUMI has cray-fftw; if its SIMD targeting mismatches the nodes, fall back to
  `-DGMX_BUILD_OWN_FFTW=ON` (the aurum2 fix — never fftpack). MPI: start thread-MPI (OFF above);
  a cray-mpich + GPU-aware-MPI build is a separate later step if multi-node cph is ever wanted.
- **Toolchain:** LUMI modules (`LUMI/xx.yy partition/G`, `rocm/…`); GROMACS needs ROCm ≥ 5.2 and
  recommends recent — use the newest LUMI-blessed ROCm (open question Q1). Compile on login nodes
  (no GPU needed); run via SLURM `standard-g`/`small-g`, `--gpus-per-node=1`, and pin with
  `ROCR_VISIBLE_DEVICES` (the HIP analogue of `CUDA_VISIBLE_DEVICES`; GROMACS's own `-gpu_id` also
  works). For perf numbers (H5) use the LUMI CPU-GCD binding recipe (each GCD has a NUMA-local
  L3 group); for correctness gates binding is irrelevant.
- **Runtime library discipline:** same class of hazard as both documented gotchas — check with
  `ldd` which `libgromacs` the binary resolves before trusting any result, and prefer in-place lib
  swaps for A/B tests (the binary's RPATH outranks `LD_LIBRARY_PATH`, as on aurum2).
- **Debug env vars** work unchanged (`GMX_CPH_DUMP_DVDL`, `GMX_CPH_DUMP_LAMBDAS`); useful AMD ones:
  `AMD_SERIALIZE_KERNEL=3` / `AMD_SERIALIZE_COPY=3` (sync-after-launch, the
  `CUDA_LAUNCH_BLOCKING` analogue) and `HSA_ENABLE_SDMA=0` when chasing async-copy suspicions.

---

## 9. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Silent-zero reciprocal potential in the H1→H2 window** (buffers allocated + copied back as zeros; dV/dλ loses the PME term with no error) | High if mis-staged | Guard flips only with H1+H2 both present (§6); dev-branch bring-up tests RF-only or carries a scratch `useGpuPme` fatal; G2 includes the "every λ-atom V_i non-zero" check |
| 2 | Wavefront 32→64: shuffle semantics / lane layout in the PME potential reduce | Medium | Pattern copied from stock `pme_gather_hip.cpp:128` which already assumes the same layout; no `_sync`/mask ported; sub-width (16-lane) shuffles identical on both widths; G2 catches any residue. NB + buffer-op pieces have **no** lane communication (§4) |
| 3 | epsfac charge-scaling convention flipped in the NB accumulation | Medium (likely but cheap) | Explicitly documented (§4 item 4); a ×138.9 error is unmissable at G1 |
| 4 | VkFFT numerics floor unknown → gate tolerance guessed wrong | Low-Medium | Measure the non-cph PME GPU-vs-CPU floor first (H2); optional rocFFT cross-build as arbiter |
| 5 | `-munsafe-fp-atomics` float atomics: summation-order nondeterminism run-to-run | Low | Same class as CUDA atomics; gates compare against CPU with FP-floor tolerances, and λ-trace gates use short horizons before Lyapunov growth |
| 6 | hipcc rejects/miscompiles the shared cph C++ (first HIP compile of those hunks) | Low | Plain C++ over the device-abstraction layer; H0 exists to surface this before kernel work |
| 7 | Multi-GCD ambition creep (DD cph resident on HIP) | Low | Out of scope by existing guard — classic path covers DD; device group-reduce/PME-rank comm are separate WPs on any backend |
| 8 | No local AMD GPU — every iteration goes through LUMI's queue | Medium (schedule) | Batch the gates; keep single-GCD `small-g` jobs short; the aurum2 experience says budget ~2× calendar overhead |
| 9 | Fork oracle not runnable on LUMI GPUs | None (by design) | Oracle chain runs through the fork-validated CPU port (§7 H0); the fork itself is only ever needed on x86 CPU, which LUMI also has if a direct re-check is wanted |
| 10 | Stock-HIP regression from cph edits | Low | H1-H3 touch only cph-gated branches (`computePotential`/`gm_charge != nullptr`); H4 includes a non-cph HIP A/B and a CUDA re-run |

---

## 10. Open questions (resolve before coding)

1. **LUMI toolchain pin:** which `LUMI/xx.yy` stack + `rocm` module to standardize on (ROCm ≥ 5.2
   required; 2026.1 was CI-tested against HIP 5.7.1/6.2.2 per install guide:1797)? Is there a
   CSC/LUMI GROMACS-2026 build recipe worth copying wholesale (they publish one per stack)?
2. **MPI scope:** does the LUMI campaign need anything beyond one thread-MPI rank per GCD per
   window? If yes (multi-GCD DD), a `GMX_MPI=ON` cray-mpich build + GPU-aware-MPI testing enters
   scope (classic path only, per the resident guard) — different effort class.
3. **FFT backend choice:** VkFFT (default) vs rocFFT on gfx90a for our ~90k-atom grids — pick by a
   quick perf+floor measurement in H0/H2; the plan assumes VkFFT.
4. **Gate tolerance for G2** — fix only after measuring the VkFFT single-precision floor (Q3).
5. **Branch/merge policy:** confirm `hip-port` branch off `master`, merge at H4, no push until
   asked (repo convention). Should the LUMI clone follow the aurum2 pattern (one clone, wrappers in
   `.git/info/exclude`, ‼️ always `git fetch` before trusting `origin/master`)?
6. **SYCL later?** LUMI also runs AdaptiveCpp; the guard message and README should say "CUDA and
   HIP" now, with SYCL explicitly listed as not implemented — confirm nobody expects ACpp-SYCL on
   LUMI instead of native HIP (native HIP is the right target: first-class in 2026, and this plan's
   file map is HIP-specific).
7. **Science gate on LUMI:** after H4, is a short titration-window comparison (λ histograms vs a
   CUDA or CPU run — the M3-class check) wanted before production, or do the numerical gates
   suffice for the umbrella campaign?
8. **MI250X packed-math opportunity** (H5 only): worth evaluating `v_pk_*` friendliness of the
   potential accumulation, or is NB kernel time already dominated elsewhere? (Defer; measure first.)

---

## 11. Bookkeeping on landing

- Append a `PORT_LOG.md` section per work package (commit → WP → bug → gate result), per repo rule.
- Update `README.md` limitation 1 (CUDA+HIP), and the "Validation" table with G1/G2/G3 rows.
- Update `GPU_LAMBDA_UPDATE_PLAN.md` L5 line: HIP done, SYCL/OpenCL remaining.
- Add the LUMI wrappers (`configure_hip.sh`, `build_hip.sh`, job templates) as cluster-local
  untracked files on LUMI, mirroring the aurum2 convention; document the LUMI tree in `CLAUDE.md`'s
  "other trees" table.
