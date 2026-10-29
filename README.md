# jaxweb

NEXUS and NEXUS+ cosmic web classification ([Cautun et al. 2013](https://arxiv.org/abs/1209.2043)) on many GPUs with JAX.

jaxweb is a from-scratch, multi-GPU rewrite of [PyCosmoMMF](https://github.com/James11222/PyCosmoMMF). It is built so that a 2048³ single-precision density cube runs on one node of 4× A100 80 GB, and it scales to more GPUs and nodes with the same code.

Status: multi-scale NEXUS and NEXUS+ signatures. Structure tagging (clusters, filaments, walls, voids) is next.

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
```

The input must come from a periodic box, and `nx` and `ny` must be divisible by the number of GPUs.

On GPUs, set these before starting Python (see `benchmarks/run_a100_node.slurm`):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
```

## How it works

- Real-space fields are split along x into one slab per GPU; k-space fields are split along k_y. The 3D FFT is a local 2D FFT, one `all_to_all`, and a local 1D FFT.
- The Hessian H_ij = −k_i k_j R² f̂ is built from three partial inverse transforms over x only (k_x to the power 0, 1, 2). Each takes one chunked all-to-all. The six components are then made a few x-planes at a time, straight into the eigenvalues, signatures and running max, so they never exist at full size.
- Eigenvalues use the closed-form Smith (1961) solver, the same as PyCosmoMMF.

Memory per GPU at 2048³ on 4 GPUs, from XLA's buffer assignment (`benchmarks/memory_estimate.py`):

| Algorithm | Peak per GPU |
| --- | --- |
| NEXUS | 57 GiB |
| NEXUS+ | 65 GiB |
| NEXUS+ with `offload=True` (log-spectrum kept in host memory) | 57 GiB |

`offload=None` (the default) turns host offload on when the estimate does not fit in 90% of device memory.

## Tests

```bash
pytest
```

`tests/conftest.py` simulates 4 CPU devices, so the sharding and all-to-alls are tested without a GPU. The parity tests compare smoothing, Hessians and signatures with PyCosmoMMF.
