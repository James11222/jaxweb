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
from jax.experimental.layout import Layout, with_layout_constraint

from .mesh import AXIS, CHUNK_ELEMENTS, KSPACE_SPEC, REAL_SPEC

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


def _rows_per_chunk(n_rows, row_elements, chunk_elements):
    """The largest divisor of ``n_rows`` whose rows fit in ``chunk_elements``."""
    return largest_divisor_at_most(n_rows, chunk_elements // row_elements)


def local_rfft3(x, chunk_elements=CHUNK_ELEMENTS, pre=None):
    """
    Forward 3D rfft: real x-slab ``(nx/P, ny, nz)`` to k_y-slab ``(nx, ny/P, nz//2+1)``.

    Done in chunks, so there is no full-slab FFT work area or transpose: the
    (z, y) transforms a few x-planes at a time into one buffer, then
    ``local_forward_x``. ``pre``, if given, is applied to each real chunk
    first (``log10``, say), so the transformed field is never stored.
    """
    nx_local, ny, nz = x.shape
    planes = _rows_per_chunk(nx_local, ny * nz, chunk_elements)

    def body(c, a):
        start = c * planes
        chunk = lax.dynamic_slice_in_dim(x, start, planes, axis=0)
        if pre is not None:
            chunk = pre(chunk)
        chunk = jnp.fft.fft(jnp.fft.rfft(chunk, axis=2), axis=1)
        return lax.dynamic_update_slice_in_dim(a, _row_major(chunk), start, axis=0)

    # The initial value is overwritten; see local_partial_inverse_x for why
    # it is built from the input.
    a = jnp.broadcast_to((x[0, 0, 0] * 0).astype(jnp.complex64), (nx_local, ny, nz // 2 + 1))
    return local_forward_x(lax.fori_loop(0, nx_local // planes, body, a), chunk_elements)


def local_forward_x(a, chunk_elements=CHUNK_ELEMENTS, scale=None):
    """
    The last step of ``local_rfft3``: an x-slab transformed in (y, z), shape
    ``(nx/P, ny, nz//2+1)``, to the k_y-slab ``(nx, ny/P, nz//2+1)``.

    A few k_y columns at a time: all-to-all, FFT over x, times ``scale`` if
    given, written into the output.
    """
    nx_local, ny, nzh = a.shape
    n_dev = lax.axis_size(AXIS)
    nx, ny_local = nx_local * n_dev, ny // n_dev
    columns = _rows_per_chunk(ny_local, nx * nzh, chunk_elements)
    a = a.reshape(nx_local, n_dev, ny_local, nzh)

    def body(c, out):
        start = c * columns
        # k_y = d * ny_local + start + j, for every destination device d.
        block = _row_major(lax.dynamic_slice_in_dim(a, start, columns, axis=2))
        block = lax.all_to_all(block, AXIS, split_axis=1, concat_axis=0, tiled=True)
        block = jnp.fft.fft(block.reshape(nx, columns, nzh), axis=0)
        if scale is not None:
            block = block * scale
        return lax.dynamic_update_slice_in_dim(out, _row_major(block), start, axis=1)

    out = jnp.broadcast_to(a[0, 0, 0, 0] * 0, (nx, ny_local, nzh))
    return lax.fori_loop(0, ny_local // columns, body, out)


def local_inverse_x(a, chunk_elements=CHUNK_ELEMENTS, multiplier=None):
    """
    The first step of ``local_irfft3``: the k_y-slab ``(nx, ny/P, nz//2+1)``,
    times ``multiplier(kx_index, ky_index, kz_index)`` if given (a filter,
    say), to an x-slab still spectral in (k_y, k_z), ``(nx/P, ny, nz//2+1)``.
    """
    nx, ny_local, nzh = a.shape
    n_dev = lax.axis_size(AXIS)
    if multiplier is None:

        def multiplier(kx_index, ky_index, kz_index):
            return jnp.float32(1.0)

    columns = _rows_per_chunk(ny_local, nx * nzh, chunk_elements)
    return local_partial_inverse_x(a, [multiplier], n_dev, ny_local // columns)[0]


def local_irfft3(a, nz, chunk_elements=CHUNK_ELEMENTS):
    """
    Inverse of ``local_rfft3``: k_y-slab back to a real x-slab with ``nz`` planes.

    Chunked like ``local_rfft3``: ``local_inverse_x``, then the (y, z)
    transforms a few x-planes at a time.
    """
    b = local_inverse_x(a, chunk_elements)
    nx_local, ny, _ = b.shape
    planes = _rows_per_chunk(nx_local, ny * nz, chunk_elements)

    def body(c, out):
        start = c * planes
        chunk = lax.dynamic_slice_in_dim(b, start, planes, axis=0)
        chunk = jnp.fft.irfft2(chunk, s=(ny, nz), axes=(1, 2))
        return lax.dynamic_update_slice_in_dim(out, _row_major(chunk), start, axis=0)

    out = jnp.broadcast_to(jnp.real(b[0, 0, 0]) * 0, (nx_local, ny, nz))
    return lax.fori_loop(0, nx_local // planes, body, out)


def local_map_real_space(b, nz, fn, chunk_elements=CHUNK_ELEMENTS):
    """
    Apply ``fn`` in real space to an x-slab held spectral in (k_y, k_z).

    ``b`` is ``(nx/P, ny, nz//2+1)`` as made by ``local_inverse_x``. A few
    x-planes at a time: inverse (y, z) transform, ``fn``, forward (y, z)
    transform, written back in place, so the real field is never stored.
    Returns the new x-slab, ready for ``local_forward_x``, and this device's
    sum of ``fn``'s output.
    """
    nx_local, ny, _ = b.shape
    planes = _rows_per_chunk(nx_local, ny * nz, chunk_elements)

    def body(c, carry):
        b, total = carry
        start = c * planes
        chunk = lax.dynamic_slice_in_dim(b, start, planes, axis=0)
        chunk = fn(jnp.fft.irfft2(chunk, s=(ny, nz), axes=(1, 2)))
        total = total + jnp.sum(chunk)
        chunk = jnp.fft.fft(jnp.fft.rfft(chunk, axis=2), axis=1)
        return lax.dynamic_update_slice_in_dim(b, _row_major(chunk), start, axis=0), total

    total = jnp.real(b[0, 0, 0]) * 0
    return lax.fori_loop(0, nx_local // planes, body, (b, total))


def _row_major(x):
    """
    ``x`` pinned to the default row-major layout.

    XLA GPU stores a loop's buffers in whatever layout suits the ops inside
    the loop. When a loop body transposes a chunk (for an FFT over a
    non-minor axis, or after an all-to-all), XLA may store the whole loop
    buffer transposed instead and copy it once outside the loop: a full
    slab of extra memory. Pinning the chunk keeps the transpose inside the
    loop, at chunk size.
    """
    return with_layout_constraint(x, Layout(tuple(range(x.ndim))))


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
        # Pinned so the x-transform's transpose is done per chunk (see _row_major).
        block = _row_major(lax.dynamic_slice_in_dim(spec, start, ny_chunk, axis=1))
        ky_index = local_ky_index(ny_local, start, ny_chunk)[None, :, None]
        block = jnp.stack([block * m(kx_index, ky_index, kz_index) for m in multipliers], axis=1)
        block = jnp.fft.ifft(block, axis=0)  # (nx, M, ny_chunk, nzh)
        # Device d receives x-planes d*nx_local... from every source device s,
        # concatenated in order of s: global k_y = s*ny_local + start + j.
        block = lax.all_to_all(block, AXIS, split_axis=0, concat_axis=1, tiled=True)
        block = block.reshape(nx_local, n_dev, n_out, ny_chunk, nzh).transpose(2, 0, 1, 3, 4)
        return lax.dynamic_update_slice_in_dim(out, _row_major(block), start, axis=3)

    # Every element is overwritten, so the initial value does not matter. It
    # is built from ``spec`` (times 0) only so that XLA allocates this buffer
    # after ``spec`` exists: an independent zeros array may be scheduled
    # early and overlap the NEXUS+ smoothing transients.
    out = jnp.broadcast_to(spec[0, 0, 0] * 0, (n_out, nx_local, n_dev, ny_local, nzh))
    out = lax.fori_loop(0, n_chunks, body, out)
    return out.reshape(n_out, nx_local, n_dev * ny_local, nzh)


def rfft3(x, mesh, chunk_elements=CHUNK_ELEMENTS):
    """Forward 3D rfft of an x-sharded real field; the result is k_y-sharded."""
    fn = jax.shard_map(
        lambda y: local_rfft3(y, chunk_elements),
        mesh=mesh,
        in_specs=REAL_SPEC,
        out_specs=KSPACE_SPEC,
    )
    return jax.jit(fn)(x)


def irfft3(a, mesh, nz, chunk_elements=CHUNK_ELEMENTS):
    """Inverse of ``rfft3``; ``nz`` is the real-space length of the last axis."""
    fn = jax.shard_map(
        lambda b: local_irfft3(b, nz, chunk_elements),
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
