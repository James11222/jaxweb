"""Connected clumps of a sharded boolean mask, with per-clump δ sums.

Each device labels its own x-slab (CuPy on GPUs, SciPy on CPU devices, both
26-connected), all devices at once, one host thread each. Clumps that touch across a slab boundary are then joined: the
last x-plane of every slab is compared with the first plane of the next one,
every pair of touching labels becomes an edge, and a connected-components
pass over those edges on the host maps every local label to its clump.

Clumps are not joined across the faces of the box, to match PyCosmoMMF
(scikit-image's labelling is not periodic).
"""

from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
from jax import lax

from .fft import largest_divisor_at_most
from .mesh import CHUNK_ELEMENTS, as_sharded, real_sharding

_STRUCTURE = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity

# Largest piece labelled at once (see ``clump_mean_overdensity``).
LABEL_ELEMENTS = 2**30


def _label_shard(mask):
    """Label one device's uint8 mask in place on that device: ``(labels, n)``."""
    device = next(iter(mask.devices()))
    if device.platform == "gpu":
        import cupy as cp
        import cupyx.scipy.ndimage

        with cp.cuda.Device(device.local_hardware_id):
            labels, n = cupyx.scipy.ndimage.label(
                cp.from_dlpack(mask).astype(bool), structure=_STRUCTURE
            )
            return jax.dlpack.from_dlpack(labels.astype(cp.int32, copy=False)), int(n)
    labels, n = scipy.ndimage.label(np.asarray(mask), structure=_STRUCTURE)
    return jax.device_put(labels.astype(np.int32), device), int(n)


@functools.partial(jax.jit, static_argnums=2)
def _segment_sums(labels, rho, num_segments):
    """
    float32 sum of δ = ρ - 1 and int32 voxel count per label (index 0 is
    background), a few x-planes at a time so no slab-sized temporary is made.
    """
    nx, ny, nz = labels.shape
    planes = largest_divisor_at_most(nx, CHUNK_ELEMENTS // (ny * nz))

    def body(c, carry):
        sums, counts = carry
        lab = lax.dynamic_slice_in_dim(labels, c * planes, planes, axis=0).ravel()
        delta = lax.dynamic_slice_in_dim(rho, c * planes, planes, axis=0).ravel() - 1.0
        sums = sums + jax.ops.segment_sum(delta, lab, num_segments=num_segments)
        counts = counts + jax.ops.segment_sum(jnp.ones_like(lab), lab, num_segments=num_segments)
        return sums, counts

    init = (jnp.zeros(num_segments, jnp.float32), jnp.zeros(num_segments, jnp.int32))
    return lax.fori_loop(0, nx // planes, body, init)


def _bucket(n):
    """Round ``n`` up to a power of two, so ``_segment_sums`` compiles a few sizes only."""
    return 1 << max(4, int(n).bit_length())


def _touching_pairs(last_plane, first_plane):
    """Label pairs ``(a, b)`` that touch across a slab boundary under 26-connectivity."""
    ny, nz = last_plane.shape
    pairs = []
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            a = last_plane[max(0, -dy) : ny - max(0, dy), max(0, -dz) : nz - max(0, dz)]
            b = first_plane[max(0, dy) : ny + min(0, dy), max(0, dz) : nz + min(0, dz)]
            both = (a > 0) & (b > 0)
            pairs.append(np.stack([a[both], b[both]], axis=1))
    return np.unique(np.concatenate(pairs), axis=0)


def _label_slab(mask, rho):
    """
    Label one device's slab, in pieces of at most LABEL_ELEMENTS voxels (a
    2048³ slab on one of 4 GPUs has exactly 2^31, where int32 indexing can
    break). Per piece, along x: ``(sums, counts, first plane, last plane, n)``
    on the host. Pieces are stitched exactly like slabs.
    """
    nx, ny, nz = mask.shape
    planes = largest_divisor_at_most(nx, LABEL_ELEMENTS // (ny * nz))
    pieces = []
    for start in range(0, nx, planes):
        piece = mask if planes == nx else mask[start : start + planes]
        labels, n = _label_shard(piece)
        rho_piece = rho if planes == nx else rho[start : start + planes]
        s, c = _segment_sums(labels, rho_piece, _bucket(n + 1))
        pieces.append(
            (
                np.asarray(s)[1 : n + 1],
                np.asarray(c)[1 : n + 1],
                np.asarray(labels[0]),
                np.asarray(labels[-1]),
                n,
            )
        )
        del labels, piece, rho_piece
    return pieces


def clump_mean_overdensity(mask, rho, mesh):
    """
    Mean δ of every 26-connected clump of ``mask``, across all slabs.

    Args:
        mask: x-sharded boolean or uint8 array ``(nx, ny, nz)``.
        rho: float32 δ+1 = ρ/⟨ρ⟩, same shape (δ is formed per slab, on the fly).
        mesh: the device mesh; both arrays are laid out as x-slabs on it.

    Returns:
        float32 NumPy array, one mean per clump (in no particular order).
    """
    if jax.process_count() > 1:
        msg = "clump labelling across several processes is not implemented yet"
        raise NotImplementedError(msg)

    mask = as_sharded(mask, real_sharding(mesh), jnp.uint8)
    rho = as_sharded(rho, real_sharding(mesh), jnp.float32)
    rho_shards = {s.index[0].start or 0: s.data for s in rho.addressable_shards}

    shards = sorted(mask.addressable_shards, key=lambda s: s.index[0].start or 0)
    # One thread per device, so the devices label at the same time: each
    # thread mostly waits on its device, with the GIL released.
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        slabs = list(
            pool.map(lambda s: _label_slab(s.data, rho_shards[s.index[0].start or 0]), shards)
        )
    pieces = [piece for slab in slabs for piece in slab]
    sums, counts, first, last, n_local = map(list, zip(*pieces, strict=True))

    offsets = np.concatenate([[0], np.cumsum(n_local)])
    n_total = int(offsets[-1])
    if n_total == 0:
        return np.zeros(0, dtype=np.float32)

    # Label l of slab d is node offsets[d] + l - 1.
    edges = [np.zeros((0, 2), dtype=np.int64)]
    for d in range(len(n_local) - 1):  # consecutive pieces along x
        pairs = _touching_pairs(last[d], first[d + 1]).astype(np.int64)
        if len(pairs):
            edges.append(np.stack([pairs[:, 0] + offsets[d], pairs[:, 1] + offsets[d + 1]], 1) - 1)
    edges = np.concatenate(edges)
    graph = scipy.sparse.coo_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
        shape=(n_total, n_total),
    )
    n_clumps, clump_of = scipy.sparse.csgraph.connected_components(graph, directed=False)

    clump_sums = np.zeros(n_clumps, dtype=np.float32)
    np.add.at(clump_sums, clump_of, np.concatenate(sums).astype(np.float32))
    counts_all = np.zeros(n_clumps, dtype=np.int64)
    np.add.at(counts_all, clump_of, np.concatenate(counts).astype(np.int64))
    return clump_sums / counts_all.astype(np.float32)
