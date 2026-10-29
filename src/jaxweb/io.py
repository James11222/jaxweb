"""Sharded loading of density cubes: each process reads only its own slabs."""

from __future__ import annotations

import jax
import numpy as np

from .mesh import check_shape, make_mesh, real_sharding


def load_density(path, mesh=None, dataset=None):
    """
    Load a 3D density cube as an x-sharded float32 JAX array.

    ``.npy`` files are memory-mapped and ``.h5``/``.hdf5`` files (give the
    ``dataset`` name) are read by hyperslab, so each process reads only the
    x-planes of its own devices and no process ever holds the whole cube.
    """
    mesh = make_mesh() if mesh is None else mesh
    path = str(path)
    if path.endswith((".h5", ".hdf5")):
        import h5py

        if dataset is None:
            msg = "pass dataset= to read an HDF5 file"
            raise ValueError(msg)
        handle = h5py.File(path, "r")
        source = handle[dataset]
    else:
        handle = None
        source = np.load(path, mmap_mode="r")

    try:
        shape = tuple(source.shape)
        check_shape(shape, mesh)
        return jax.make_array_from_callback(
            shape,
            real_sharding(mesh),
            lambda index: np.asarray(source[index], dtype=np.float32),
        )
    finally:
        if handle is not None:
            handle.close()
