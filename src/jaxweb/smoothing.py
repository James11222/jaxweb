"""Gaussian (NEXUS) and log-Gaussian (NEXUS+) smoothing in k-space."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from .fft import (
    local_forward_x,
    local_inverse_x,
    local_irfft3,
    local_ky_index,
    local_map_real_space,
    local_rfft3,
    wavenumbers,
)
from .mesh import AXIS, CHUNK_ELEMENTS, REAL_SPEC, check_shape

NEXUS = "NEXUS"
NEXUSPLUS = "NEXUSPLUS"
ALGORITHMS = (NEXUS, NEXUSPLUS)


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


def smooth(density, R, mesh, algorithm=NEXUSPLUS, eps=1e-16):
    """
    The smoothed real-space field at scale ``R`` (voxels), x-sharded.

    ``NEXUS`` smooths ρ with a Gaussian, ``NEXUSPLUS`` smooths log10(ρ) and
    renormalises the mass. Mainly for checks and plots: ``maximum_signature``
    never builds this field for NEXUS.
    """
    check_shape(density.shape, mesh)
    shape = density.shape
    R = jnp.float32(R)

    def local(rho, R):
        field = rho + jnp.float32(eps)
        if algorithm == NEXUSPLUS:
            total = lax.psum(jnp.sum(field), AXIS)
            spec = local_loggauss_spectrum(local_rfft3(jnp.log10(field)), total, R, shape)
        else:
            spec = local_rfft3(field)
            spec = spec * local_gaussian_block(spec.shape, shape, R)
        return local_irfft3(spec, shape[2])

    fn = jax.shard_map(local, mesh=mesh, in_specs=(REAL_SPEC, P()), out_specs=REAL_SPEC)
    return jax.jit(fn)(density, R)
