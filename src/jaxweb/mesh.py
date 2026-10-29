"""Device mesh, multi-process start-up and the shardings every module uses.

jaxweb uses a 1D mesh with one axis, ``"x"``, over every GPU of every process.

* Real-space fields of shape ``(nx, ny, nz)`` are split along x: each device
  holds ``nx / P`` consecutive x-planes (``REAL_SPEC``).
* Signature arrays of shape ``(3, nx, ny, nz)`` are split the same way
  (``SIG_SPEC``).
* k-space fields (the rfft half-spectrum, ``(nx, ny, nz // 2 + 1)``) are split
  along k_y (``KSPACE_SPEC``), so the x-transform is local.
"""

from __future__ import annotations

import os

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

AXIS = "x"

REAL_SPEC = P(AXIS, None, None)
KSPACE_SPEC = P(None, AXIS, None)
SIG_SPEC = P(None, AXIS, None, None)

_MULTI_PROCESS_ENV = ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")


def initialize(**kwargs):
    """
    Start JAX's multi-process runtime when the job runs more than one process.

    Under SLURM or MPI with more than one task this calls
    ``jax.distributed.initialize(**kwargs)``, which finds the coordinator from
    the environment. With a single process (one node, any number of GPUs) it
    does nothing, so the same script runs everywhere. Call it before any other
    JAX call.
    """
    n_tasks = max(int(os.environ.get(name, "1")) for name in _MULTI_PROCESS_ENV)
    if n_tasks > 1 or kwargs:
        jax.distributed.initialize(**kwargs)


def make_mesh(devices=None):
    """A 1D ``Mesh`` over ``devices`` (default: every device of every process)."""
    devices = jax.devices() if devices is None else list(devices)
    # Auto axes: shard_map reshards inputs that arrive with another layout
    # (a NumPy array, say) instead of raising.
    return jax.make_mesh(
        (len(devices),), (AXIS,), axis_types=(jax.sharding.AxisType.Auto,), devices=devices
    )


def real_sharding(mesh):
    return NamedSharding(mesh, REAL_SPEC)


def kspace_sharding(mesh):
    return NamedSharding(mesh, KSPACE_SPEC)


def sig_sharding(mesh):
    return NamedSharding(mesh, SIG_SPEC)


def check_shape(shape, mesh):
    """Raise ``ValueError`` unless a cube of ``shape`` can be split over ``mesh``."""
    if len(shape) != 3:
        msg = f"jaxweb works on 3D fields, got shape {shape}"
        raise ValueError(msg)
    nx, ny, nz = shape
    n_dev = mesh.shape[AXIS]
    if nx % n_dev or ny % n_dev:
        msg = (
            f"nx = {nx} and ny = {ny} must both be divisible by the number of "
            f"devices ({n_dev}): real space is split along x and k-space along k_y"
        )
        raise ValueError(msg)
    if nz % 2:
        msg = f"nz = {nz} must be even for the real-to-complex FFT"
        raise ValueError(msg)
