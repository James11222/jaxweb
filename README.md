# jaxweb

NEXUS and NEXUS+ cosmic web classification ([Cautun et al. 2013](https://arxiv.org/abs/1209.2043)) on many GPUs with JAX.

jaxweb is a from-scratch, multi-GPU rewrite of [PyCosmoMMF](https://github.com/James11222/PyCosmoMMF). It is built so that a 2048³ single-precision density cube runs on one node of 4× A100 80 GB, and it scales to more GPUs and nodes with the same code.

Status: multi-scale NEXUS and NEXUS+ signatures and cluster / filament / wall / void tagging on one node (any number of GPUs). Multi-node tagging is next.

## Install

```bash
pip install -e ".[gpu]"     # JAX with CUDA 12, and CuPy
pip install -e ".[test]"    # pytest, scipy, and PyCosmoMMF for the parity tests
```

## Use

```python
import jaxweb

jaxweb.initialize()          # starts jax.distributed under SLURM/MPI; no-op for one process
mesh = jaxweb.make_mesh()    # 1D mesh over every GPU

rho = jaxweb.load_density("density.npy", mesh)   # δ+1 = ρ/⟨ρ⟩; each process reads only its slabs
Rs = [2 ** (n / 2) for n in range(9)]            # smoothing scales in voxels
sigmax = jaxweb.maximum_signature(rho, Rs, algorithm="NEXUSPLUS", mesh=mesh)
# sigmax: float32 (3, nx, ny, nz), x-sharded: cluster, filament, wall signatures

structure, summary = jaxweb.tag_structures(rho, sigmax, overdensity_threshold=370, mesh=mesh)
# structure: uint8 (nx, ny, nz), x-sharded: 1 cluster, 2 filament, 3 wall, 4 void
# summary: thresholds, dM²/dlog S curves, volume and mass fractions (PyCosmoMMF's keys)
```

NEXUS+ takes log10 ρ, so empty voxels need a floor: by default ρ is clipped to at least `jaxweb.density_floor(rho, mesh)`, a tenth of the 1% quantile of the positive voxels (pass `floor=` to override). PyCosmoMMF adds `1e-16` instead, which puts empty voxels 16 decades down and gives spurious walls around every empty region.

The input must come from a periodic box, and `nx` and `ny` must be divisible by the number of GPUs.

On GPUs, set these before starting Python (see `benchmarks/run_a100_node.slurm`):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
```

## How it works

- Real-space fields are split along x into one slab per GPU; k-space fields are split along k_y. The 3D FFT is a local 2D FFT, one `all_to_all`, and a local 1D FFT, each done a few planes or columns at a time so no full-slab FFT work area or transpose is ever made. The NEXUS+ smoothing (inverse FFT, 10^x, forward FFT) is fused into one such pass.
- The Hessian H_ij = −k_i k_j R² f̂ is built from three partial inverse transforms over x only (k_x to the power 0, 1, 2). Each takes one chunked all-to-all. The six components are then made a few x-planes at a time, straight into the eigenvalues, signatures and running max, so they never exist at full size.
- Eigenvalues use the closed-form Smith (1961) solver, the same as PyCosmoMMF.
- Tagging follows PyCosmoMMF's `calc_structure_bools` step for step. Clumps are labelled per GPU (CuPy; SciPy on CPU devices) in pieces of at most 2³⁰ voxels, and clumps that touch across piece boundaries are joined on the host with a connected-components pass over the touching label pairs. Like PyCosmoMMF, clumps are not joined across the box faces.

Measured on one node of 4× A100 80 GB (Della), 2048³, 9 scales, with the caller keeping the density on the GPUs (`benchmarks/run_a100_node.slurm`):

| Algorithm | Peak per GPU | Signatures |
| --- | --- | --- |
| NEXUS | 65.3 GiB | 13 s |
| NEXUS+ with `offload=True` (log-spectrum kept in host memory) | 65.3 GiB | 46 s |
| NEXUS+ with `offload=False` | 73.3 GiB | 25 s |

`offload=None` (the default) turns host offload on when NEXUS+ would need more than 90% of the memory XLA may use (`XLA_PYTHON_CLIENT_MEM_FRACTION` of the device), which it does at 2048³ on 4× 80 GB. `benchmarks/memory_estimate.py` gives the same numbers from XLA's buffer assignment without a GPU.

Tagging needs about 40 GiB per GPU at 2048³ on 4 GPUs (density, signatures and a few masks), plus CuPy's labelling workspace for one 2³⁰-voxel piece.

## Tests

```bash
pytest
```

`tests/conftest.py` simulates 4 CPU devices, so the sharding and all-to-alls are tested without a GPU. The parity tests compare smoothing, Hessians, signatures and the tagging (thresholds and `structure_filter`) with PyCosmoMMF, and clump labelling with scikit-image.
