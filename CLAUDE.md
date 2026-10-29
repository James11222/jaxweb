# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

`jaxweb` computes the NEXUS / NEXUS+ multiscale morphology filters (Cautun et al. 2013) on many GPUs with JAX. It is a rewrite of PyCosmoMMF (`James11222/PyCosmoMMF`), which is the reference for correctness. Design target: a 2048³ float32 cube on one node of 4× A100 80 GB; portable to bigger clusters and multiple nodes.

## Commands

```bash
pip install -e ".[test]"          # CPU only; ".[gpu]" adds CUDA JAX + CuPy
pytest                            # 4 simulated CPU devices (tests/conftest.py)
ruff check src tests benchmarks && ruff format src tests benchmarks
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
    python benchmarks/memory_estimate.py 2048     # per-device memory, no allocation
```

## Conventions

- Always float32 / complex64. Input is δ+1 = ρ/⟨ρ⟩ from a periodic box. Smoothing scales are in voxels; wavenumbers are `2π · fftfreq(n)` with the Nyquist entry at −π, as in PyCosmoMMF.
- NEXUS+ clips ρ to at least `density_floor` (a tenth of the 1% quantile of the positive voxels) before log10, instead of PyCosmoMMF's `ρ + 1e-16`: with empty voxels at log10 = −16 the wall dM² curve turns bimodal and the wall threshold lands on the spurious peak. Parity tests pass `floor=cube.min()`, a no-op clip, to compare with PyCosmoMMF.
- 1D mesh, axis `"x"` (`mesh.py`). Real space `P("x", None, None)`, k-space (rfft half-spectrum) `P(None, "x", None)`, signatures `(3, nx, ny, nz)` with `P(None, "x", None, None)`.
- Functions named `local_*` run inside `jax.shard_map` on one device's block.
- Memory is the main constraint. Never build all six Hessian components at full size, never materialise `(…, 3, 3)` tensors, and keep loops chunked (`CHUNK_ELEMENTS`). `tests/test_memory.py` guards the per-device budget; run `benchmarks/memory_estimate.py 2048` after changing `fft.py`, `hessian.py` or `signatures.py`.
- Loop buffers built inside `shard_map` must be device-varying and should depend on their input (see `local_partial_inverse_x`), otherwise XLA may share or hoist the initialiser and add a full slab of memory.
- In a chunked loop, pin with `fft._row_major` any chunk that is transposed on its way in or out (an FFT over a non-minor axis, an all-to-all): the slice of a loop-invariant input and the chunk written into the loop buffer. Otherwise XLA GPU stores the whole buffer transposed and copies it outside the loop, a full slab per copy. The CPU compiler does not do this, so check memory on a GPU (`benchmarks/memory_estimate.py` runs there too) after touching a loop.
- Off-diagonal Hessian factors use the split `k = k_odd + k_nyquist` (`wavenumbers(..., part=...)`) so that the half-spectrum `irfft` matches PyCosmoMMF's real part of a full complex IFFT on the Nyquist planes. See the `hessian.py` docstring.
- Keep `eigen.eigvalsh_3x3_sym` in sync with PyCosmoMMF's `_eigvalsh_3x3_sym`, and the helpers at the top of `tagging.py` in sync with PyCosmoMMF's `tagging.py` (same thresholds and selection rules). One deliberate difference: the default cluster-signature grid is `geomspace(0.1, 60, 20)`, PyCosmoMMF's is `(0.1, 30, 10)`; tests pass PyCosmoMMF's grid via `cluster_thresholds=`.
- Tagging (`tagging.py`, `labels.py`) must match PyCosmoMMF's `structure_filter` and thresholds exactly on the test cubes. Pass the whole `(3, …)` signature array into jitted functions and index inside: `sigmax[i]` outside jit copies a full field. Clump labelling runs per device outside jit (CuPy via DLPack on GPU, SciPy on CPU) in pieces of at most `LABEL_ELEMENTS`; multi-process stitching is not implemented yet.
- Every source file starts with `from __future__ import annotations` (ruff enforces it).
