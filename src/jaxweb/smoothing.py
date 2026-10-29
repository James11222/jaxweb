"""Gaussian (NEXUS) and log-Gaussian (NEXUS+) smoothing in k-space."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec as P

from .fft import (
    largest_divisor_at_most,
    local_forward_x,
    local_inverse_x,
    local_irfft3,
    local_ky_index,
    local_map_real_space,
    local_rfft3,
    wavenumbers,
)
from .mesh import AXIS, CHUNK_ELEMENTS, REAL_SPEC, as_sharded, check_shape, real_sharding

NEXUS = "NEXUS"
NEXUSPLUS = "NEXUSPLUS"
ALGORITHMS = (NEXUS, NEXUSPLUS)

# NEXUS+ clips ρ to at least FLOOR_FACTOR times the FLOOR_QUANTILE quantile
# of the positive voxels before taking log10 (see ``density_floor``).
FLOOR_QUANTILE = 0.01
FLOOR_FACTOR = 0.1


def _make_count_in(mesh):
    """
    Jitted voxel counts of ``0 < ρ <= t``, one int32 per x-chunk and device
    (int32 cannot hold the count of a 2048³ slab); sum them on the host.
    """

    def local(rho, t):
        nx, ny, nz = rho.shape
        planes = largest_divisor_at_most(nx, CHUNK_ELEMENTS // (ny * nz))

        def count(c):
            r = lax.dynamic_slice_in_dim(rho, c * planes, planes, axis=0)
            return jnp.sum((r > 0) & (r <= t), dtype=jnp.int32)

        return lax.map(count, jnp.arange(nx // planes))

    fn = jax.shard_map(local, mesh=mesh, in_specs=(REAL_SPEC, P()), out_specs=P(AXIS))
    return jax.jit(fn)


def density_floor(density, mesh, quantile=FLOOR_QUANTILE, factor=FLOOR_FACTOR):
    """
    The NEXUS+ density floor: ``factor`` times the ``quantile`` quantile of
    the positive voxels of ``density``, as a float32.

    log10 ρ needs ρ > 0, but gridded particle data has empty voxels. Clipping
    them to a floor just below the bulk of the distribution keeps them from
    sitting many decades below every occupied voxel (where an additive
    ``eps = 1e-16`` puts them), which otherwise creates spurious walls around
    every empty region. A quantile is not moved by a few outlier voxels.

    The quantile is exact (NumPy's ``method="inverted_cdf"``): a bisection on
    the float32 bit patterns, whose order is that of the positive floats, with
    one chunked counting pass over the cube per step (31 steps).
    """
    check_shape(density.shape, mesh)
    rho = as_sharded(density, real_sharding(mesh), jnp.float32)
    count_fn = _make_count_in(mesh)

    def count(bits):
        t = np.array(bits, dtype=np.int32).view(np.float32)
        return int(np.asarray(count_fn(rho, t)).sum(dtype=np.int64))

    lo, hi = 0, int(np.array(np.finfo(np.float32).max).view(np.int32))
    n_positive = count(hi)
    if n_positive == 0:
        msg = "the density has no positive voxel, so NEXUSPLUS has no density floor"
        raise ValueError(msg)
    k = max(1, int(np.ceil(quantile * n_positive)))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(mid) >= k:
            hi = mid
        else:
            lo = mid
    return np.float32(factor) * np.array(hi, dtype=np.int32).view(np.float32)


def gaussian(kx_index, ky_index, kz_index, shape, R):
    """``exp(-k² R² / 2)`` on the given FFT indices (broadcast), float32."""
    nx, ny, nz = shape
    k2 = (
        wavenumbers(kx_index, nx) ** 2
        + wavenumbers(ky_index, ny) ** 2
        + wavenumbers(kz_index, nz) ** 2
    )
    return jnp.exp(-k2 * (jnp.float32(0.5) * R * R))


def local_gaussian_block(spec_shape, shape, R):
    """The Gaussian filter on this device's whole k-space block."""
    nx, ny_local, nzh = spec_shape
    return gaussian(
        jnp.arange(nx)[:, None, None],
        local_ky_index(ny_local)[None, :, None],
        jnp.arange(nzh)[None, None, :],
        shape,
        R,
    )


def local_loggauss_spectrum(log_spec, total, R, shape, chunk_elements=CHUNK_ELEMENTS):
    """
    Spectrum of the NEXUS+ smoothed field at scale ``R``.

    ``log_spec`` is the k-space block of ``log10(ρ)``. The field is smoothed
    in log space, brought back with ``10**x``, renormalised so its sum is
    ``total`` (the sum of ρ, as PyCosmoMMF does), and transformed again.

    The filter is applied on the fly, ``10**x`` a few x-planes at a time
    between the inverse and forward (y, z) transforms, and the
    renormalisation (linear) in k-space, so neither the smoothed log field
    nor the real field is ever stored: one spectrum-sized buffer besides
    the input and the output.
    """
    b = local_inverse_x(
        log_spec,
        chunk_elements,
        multiplier=lambda kx_index, ky_index, kz_index: gaussian(
            kx_index, ky_index, kz_index, shape, R
        ),
    )
    b, field_sum = local_map_real_space(
        b, shape[2], lambda x: jnp.power(jnp.float32(10.0), x), chunk_elements
    )
    return local_forward_x(b, chunk_elements, scale=total / lax.psum(field_sum, AXIS))


def resolve_floor(density, mesh, algorithm, floor):
    """The float32 floor for ``algorithm``: ``density_floor`` if ``floor`` is None, 0 for NEXUS."""
    if algorithm != NEXUSPLUS:
        return jnp.float32(0.0)
    return jnp.float32(density_floor(density, mesh) if floor is None else floor)


def smooth(density, R, mesh, algorithm=NEXUSPLUS, floor=None):
    """
    The smoothed real-space field at scale ``R`` (voxels), x-sharded.

    ``NEXUS`` smooths ρ with a Gaussian, ``NEXUSPLUS`` smooths log10(ρ),
    with ρ clipped to at least ``floor`` (default ``density_floor``), and
    renormalises the mass. Mainly for checks and plots:
    ``maximum_signature`` never builds this field for NEXUS.
    """
    check_shape(density.shape, mesh)
    shape = density.shape
    R = jnp.float32(R)
    floor = resolve_floor(density, mesh, algorithm, floor)

    def local(rho, R, floor):
        if algorithm == NEXUSPLUS:
            field = jnp.maximum(rho, floor)
            total = lax.psum(jnp.sum(field), AXIS)
            spec = local_loggauss_spectrum(local_rfft3(jnp.log10(field)), total, R, shape)
        else:
            spec = local_rfft3(rho)
            spec = spec * local_gaussian_block(spec.shape, shape, R)
        return local_irfft3(spec, shape[2])

    fn = jax.shard_map(local, mesh=mesh, in_specs=(REAL_SPEC, P(), P()), out_specs=REAL_SPEC)
    return jax.jit(fn)(density, R, floor)
