"""Slab-decomposed 3D real FFTs and the partial x-transform used by the Hessian.

Functions whose names start with ``local_`` run inside ``jax.shard_map`` on one
device's block and talk to the other devices through ``jax.lax.all_to_all``
over the mesh axis ``"x"``:

* real space: block ``(nx / P, ny, nz)``, float32, split along x;
* k-space: block ``(nx, ny / P, nz // 2 + 1)``, complex64, split along k_y.

``rfft3`` and ``irfft3`` wrap them for use on whole sharded arrays.

Wavenumbers are in radians per voxel, ``k = 2π · fftfreq(n)``, which is what
PyCosmoMMF's ``wavevectors3D`` gives for its default box. Smoothing scales are
therefore in voxels.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax

from .mesh import AXIS, KSPACE_SPEC, REAL_SPEC

TWO_PI = 2.0 * math.pi


def wavenumbers(index, n, part="full"):
    """
    float32 wavenumbers ``2π · fftfreq(n)`` at the integer FFT ``index``.

    The Nyquist index ``n / 2`` maps to ``-π`` as in ``numpy.fft.fftfreq`` and
    PyCosmoMMF. ``part="odd"`` gives the same values with the Nyquist entry
    set to 0, and ``part="nyquist"`` gives only the Nyquist entry (``-π``)
    and 0 elsewhere; the two add up to ``"full"``. The Hessian needs the split
    for its off-diagonal terms (see ``jaxweb.hessian``).
    """
    index = jnp.asarray(index)
    signed = jnp.where(index >= (n + 1) // 2, index - n, index)
    k = signed.astype(jnp.float32) * jnp.float32(TWO_PI / n)
    if part == "full":
        return k
    is_nyquist = (index == n // 2) & (n % 2 == 0)
    if part == "odd":
        return jnp.where(is_nyquist, jnp.float32(0.0), k)
    if part == "nyquist":
        return jnp.where(is_nyquist, k, jnp.float32(0.0))
    msg = f"part must be 'full', 'odd' or 'nyquist', got {part!r}"
    raise ValueError(msg)


def local_ky_index(ny_local, start=0, size=None):
    """Global k_y indices of this device's k-space block (or a slice of it)."""
    size = ny_local if size is None else size
    return lax.axis_index(AXIS) * ny_local + start + jnp.arange(size)


def local_rfft3(x):
    """Forward 3D rfft: real x-slab ``(nx/P, ny, nz)`` to k_y-slab ``(nx, ny/P, nz//2+1)``."""
    a = jnp.fft.rfft(x, axis=2)
    a = jnp.fft.fft(a, axis=1)
    a = lax.all_to_all(a, AXIS, split_axis=1, concat_axis=0, tiled=True)
    return jnp.fft.fft(a, axis=0)


def local_irfft3(a, nz):
    """Inverse of ``local_rfft3``: k_y-slab back to a real x-slab with ``nz`` planes."""
    a = jnp.fft.ifft(a, axis=0)
    a = lax.all_to_all(a, AXIS, split_axis=0, concat_axis=1, tiled=True)
    a = jnp.fft.ifft(a, axis=1)
    return jnp.fft.irfft(a, n=nz, axis=2)


def local_partial_inverse_x(spec, multipliers, n_dev, n_chunks):
    """
    ``ifft_x(m · spec)`` for each multiplier ``m``, moved to x-slabs but left
    spectral in (k_y, k_z).

    Args:
        spec: this device's k-space block ``(nx, ny/P, nz//2+1)``.
        multipliers: functions ``m(kx_index, ky_index, kz_index)`` returning
            the float32 factor to apply, broadcast over the three index
            arrays (shapes ``(nx,1,1)``, ``(1,c,1)``, ``(1,1,nz//2+1)``;
            ``ky_index`` holds global k_y indices).
        n_dev: number of devices on the mesh axis.
        n_chunks: the k_y columns of the block are processed this many at a
            time. Each chunk is multiplied, transformed and sent with one
            all-to-all straight into the output buffer, so no full-size
            temporary is made.

    Returns:
        ``(M, nx/P, ny, nz//2+1)`` complex64, one slab per multiplier: x in
        real space, k_y and k_z spectral. The results share one buffer
        (separate buffers would make XLA copy a shared zero-initialiser).
    """
    nx, ny_local, nzh = spec.shape
    nx_local = nx // n_dev
    ny_chunk = ny_local // n_chunks
    n_out = len(multipliers)
    kx_index = jnp.arange(nx)[:, None, None]
    kz_index = jnp.arange(nzh)[None, None, :]

    def body(c, out):
        start = c * ny_chunk
        block = lax.dynamic_slice_in_dim(spec, start, ny_chunk, axis=1)
        ky_index = local_ky_index(ny_local, start, ny_chunk)[None, :, None]
        block = jnp.stack([block * m(kx_index, ky_index, kz_index) for m in multipliers], axis=1)
        block = jnp.fft.ifft(block, axis=0)  # (nx, M, ny_chunk, nzh)
        # Device d receives x-planes d*nx_local... from every source device s,
        # concatenated in order of s: global k_y = s*ny_local + start + j.
        block = lax.all_to_all(block, AXIS, split_axis=0, concat_axis=1, tiled=True)
        block = block.reshape(nx_local, n_dev, n_out, ny_chunk, nzh).transpose(2, 0, 1, 3, 4)
        return lax.dynamic_update_slice_in_dim(out, block, start, axis=3)

    # Every element is overwritten, so the initial value does not matter. It
    # is built from ``spec`` (times 0) only so that XLA allocates this buffer
    # after ``spec`` exists: an independent zeros array may be scheduled
    # early and overlap the NEXUS+ smoothing transients.
    out = jnp.broadcast_to(spec[0, 0, 0] * 0, (n_out, nx_local, n_dev, ny_local, nzh))
    out = lax.fori_loop(0, n_chunks, body, out)
    return out.reshape(n_out, nx_local, n_dev * ny_local, nzh)


def rfft3(x, mesh):
    """Forward 3D rfft of an x-sharded real field; the result is k_y-sharded."""
    fn = jax.shard_map(local_rfft3, mesh=mesh, in_specs=REAL_SPEC, out_specs=KSPACE_SPEC)
    return jax.jit(fn)(x)


def irfft3(a, mesh, nz):
    """Inverse of ``rfft3``; ``nz`` is the real-space length of the last axis."""
    fn = jax.shard_map(
        lambda b: local_irfft3(b, nz),
        mesh=mesh,
        in_specs=KSPACE_SPEC,
        out_specs=REAL_SPEC,
    )
    return jax.jit(fn)(a)


def largest_divisor_at_most(n, limit):
    """The largest divisor of ``n`` that is ``<= limit`` (at least 1)."""
    limit = max(1, int(limit))
    for d in range(min(n, limit), 0, -1):
        if n % d == 0:
            return d
    return 1
