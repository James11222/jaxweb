# jaxweb: design principles and motives

Context for a Claude Code session fixing the out-of-memory error at 2048³ on one 4× A100 80 GB node. Read this together with `CLAUDE.md` in the repo and the design note: https://claude.ai/artifact/6Ct7qrcm34gQiE48RTE7WC

## Goals (from James, fixed)

- Implement NEXUS and NEXUS+ (Cautun et al. 2013) on many GPUs with JAX, with CuPy only where JAX has no equivalent (connected-component labelling).
- **Hard target: a 2048³ float32 cube on one node of 4× A100 80 GB.** That is the smallest machine James uses, so meeting it keeps the code portable to bigger clusters and multiple nodes.
- Input is always δ+1 = ρ/⟨ρ⟩ from a **periodic** box, and always **single precision** (float32 / complex64). Never introduce float64.
- **PyCosmoMMF is the reference.** Signatures agree to float32 FFT precision. Tagging thresholds and the uint8 `structure_filter` (1 cluster, 2 filament, 3 wall, 4 void) match exactly on the test cubes.

## Core design

1. **Periodic, so FFTs.** The Hessian is H_ij = −k_i k_j R² Ŵ f̂ in k-space. Real-space stencils would need halos of about 4R voxels for the Gaussian, which is the whole slab at large R.
2. **1D slab decomposition**, mesh axis `"x"`. Real space is split along x, `(nx/P, ny, nz)` per device. The rfft half-spectrum is split along k_y, `(nx, ny/P, nz/2+1)`. The 3D FFT is a local 2D FFT, one `lax.all_to_all`, and a local 1D FFT, hand-written with `jax.shard_map` (`fft.py`). Slabs work up to P = N GPUs.
3. **The Hessian never exists at full size** (`hessian.py`). Split the k_x factor off before the transpose: g_m = ifft_x(−R² k_x^m Ŵ f̂) for m = 0, 1, 2. That is 3 chunked all-to-alls instead of 6. Each component is a k_y/k_z factor times one g_m, followed by a local 2D irfft over (y, z), done a few x-planes at a time and fed straight into the eigenvalues, the signatures and the running max.
4. **Chunk everything that touches a full slab** (`CHUNK_ELEMENTS = 2**25`): the k_y loop in `local_partial_inverse_x` and the x-plane loop in `_local_update_signatures`. The tagging reductions are chunked the same way.
5. **Closed-form eigenvalues** (Smith 1961, max-abs rescaled), elementwise. There is no `(…,3,3)` tensor and no `eigh`. This is the same solver as PyCosmoMMF, so keep the two in sync.
6. **NEXUS fuses the Gaussian into the Hessian multipliers.** No smoothed field is ever stored. NEXUS+ needs a real-space round trip (smooth log10 ρ, 10^x, renormalise the mass, forward FFT), so it runs as a separate jit (`_make_loggauss_step`) before the signature jit.
7. **Host offload (`offload=`)** keeps the NEXUS+ log-spectrum in pinned host memory between scales, which saves one slab per GPU. It turns on automatically when `memory_stats()["bytes_limit"]` is too small.
8. **Tagging** mirrors PyCosmoMMF step for step. Clumps are labelled per device with CuPy in pieces of at most 2³⁰ voxels, then stitched across piece and slab faces on the host. Multi-process stitching is not written yet.

## Memory budget (per GPU, 2048³ on 4 GPUs; 1 slab = 8 GiB)

From XLA's buffer assignment on the **CPU** compiler (`benchmarks/memory_estimate.py 2048`):

| Stage | Slabs | GiB |
| --- | --- | --- |
| NEXUS signature step | 7.1 | 57 |
| NEXUS+ smoothing step (+3 slabs of running max) | 7.0 | 56 |
| NEXUS+ signature step | 8.1 | 65 |
| NEXUS+ signature step with offload | 7.1 | 57 |
| Tagging | ~5 | ~40 + CuPy workspace |

An A100 80 GB has about 74.5 GiB, minus the CUDA context. **These numbers are not GPU numbers.** They leave out cuFFT work areas, NCCL buffers and the transposes that XLA GPU adds for FFTs on non-minor axes. The real run showed that the gap matters.

## Likely causes of the OOM (check these first)

1. **The XLA allocator settings.** By default JAX preallocates 75% of the GPU (~60 GB) and caps itself there. These must be set before Python starts (see `benchmarks/run_a100_node.slurm`):
   `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`.
2. **Full-slab FFTs on non-minor axes.** `local_rfft3` / `local_irfft3` FFT a whole 8 GiB slab along axes 1 and 0. XLA GPU only transforms the minor axes, so it inserts transposes (+8 GiB each), and cuFFT may need a workspace the size of the data. These run in `_make_base` (once) and in the NEXUS+ smoothing step (every scale). Fix: chunk them like `local_partial_inverse_x`, for example the z/y transforms a few x-planes at a time and the x transform a few k_y columns at a time.
3. **`_make_base` was never budgeted.** It holds ρ, ρ+eps, log10, and FFT transients at once. Free ρ before it runs (the caller still holds `rho`), or chunk it.
4. **2³¹-element slabs.** 512 × 2048 × 2048 = 2³¹ exactly, which can break int32 indexing in XLA, cuFFT or CuPy. Keep every kernel's operand below that size.
5. **Which stage OOMed** matters: signatures (NEXUS or NEXUS+, smoothing or Hessian step) or tagging (CuPy `label`). The traceback names the jitted function.

## How to work

- Measure on the GPU, don't guess. Use `jax.local_devices()[i].memory_stats()["peak_bytes_in_use"]`, `jax.profiler.save_device_memory_profile`, and `compiled.memory_analysis()` from `step.lower(...).compile()` on the GPU. Bisect with 1024³ on 1 GPU (the same 8 GiB slab size as 2048³ on 4).
- Keep the per-device budget below ~70 GiB with the real GPU numbers, and update `tests/test_memory.py` and the README table once GPU figures exist.
- Loop buffers inside `shard_map` must be device-varying and should depend on their input (see `local_partial_inverse_x`), or XLA hoists and shares the initialiser and adds a slab.
- Don't call `sigmax[i]` outside jit. It copies a full 8 GiB field, so pass the whole array and index inside.
- Correctness guardrails: `pytest` (4 simulated CPU devices, parity with PyCosmoMMF main) must stay green. Tolerances reflect float32 FFT noise, so don't loosen them to hide a real difference.
- Preferred levers, in order: chunk more, free inputs earlier (donate, `del`), split jits so stages don't overlap, offload to host. Lower precision is not an option.

