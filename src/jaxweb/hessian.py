"""The R²-scaled Hessian H_ij = -k_i k_j R² f̂, built without six full-size arrays.

Each component carries k_x to the power 0, 1 or 2. So instead of six inverse
3D FFTs we make three partial ones, over x only:

    g_m(x, k_y, k_z) = ifft_x[ -R² k_x^m W(k) f̂(k) ],   m = 0, 1, 2

which need one all-to-all each to land on x-slabs. Each Hessian component
is then a k_y / k_z factor times one g_m, followed by a 2D inverse FFT over
(y, z) that needs no communication and can be done a few x-planes at a time.

Nyquist modes. PyCosmoMMF takes the real part of a full complex inverse FFT.
For an off-diagonal factor k_i k_j that equals using the symmetrised factor

    k_i k_j  ->  k_i' k_j' + k_i'' k_j''

where k' is k with its Nyquist entry zeroed and k'' is only the Nyquist entry
(``wavenumbers(..., part="odd" / "nyquist")``): a Nyquist mode survives only
if both factors sit at Nyquist. The half-spectrum ``irfft`` reproduces this
only when it is applied explicitly. So ``g_1`` uses k_x', and the k_x''
part, which lives on the single plane k_x = n_x/2 and is (-1)^x times that
plane after the x-transform, is carried as a small gathered plane.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from .fft import local_ky_index, local_partial_inverse_x, local_rfft3, wavenumbers
from .mesh import AXIS, REAL_SPEC, check_shape
from .smoothing import NEXUS, gaussian, local_loggauss_spectrum, resolve_floor


def local_hessian_spectra(spec, R, shape, n_dev, n_chunks, smooth=False):
    """
    The x-transformed pieces of the Hessian for this device.

    ``spec`` is the k-space block ``(nx, ny/P, nz//2+1)`` of the field to
    differentiate. With ``smooth=True`` the Gaussian filter of radius ``R`` is
    applied on the fly (NEXUS), so the smoothed spectrum is never stored.

    Returns ``(g, nyquist_plane)``: ``g`` stacks g_0, g_1, g_2 as x-slabs
    ``(3, nx/P, ny, nz//2+1)`` (``g_1`` without its k_x-Nyquist part) and the
    ``(ny, nz//2+1)`` plane ``-R² W f̂ k_x''`` at k_x = n_x/2, divided by n_x
    and gathered from all devices.
    """
    nx = shape[0]
    minus_R2 = -R * R

    def factor(kx_index, ky_index, kz_index, kx_factor):
        out = minus_R2 * kx_factor
        if smooth:
            out = out * gaussian(kx_index, ky_index, kz_index, shape, R)
        return out

    def multiplier(power):
        def fn(kx_index, ky_index, kz_index):
            if power == 0:
                kx_factor = jnp.float32(1.0)
            elif power == 1:
                kx_factor = wavenumbers(kx_index, nx, part="odd")
            else:
                kx_factor = wavenumbers(kx_index, nx) ** 2
            return factor(kx_index, ky_index, kz_index, kx_factor)

        return fn

    g = local_partial_inverse_x(spec, [multiplier(m) for m in (0, 1, 2)], n_dev, n_chunks)

    ny_local, nzh = spec.shape[1:]
    kx_nyq = nx // 2
    plane = spec[kx_nyq] * factor(
        jnp.int32(kx_nyq),
        local_ky_index(ny_local)[:, None],
        jnp.arange(nzh)[None, :],
        wavenumbers(jnp.int32(kx_nyq), nx, part="nyquist") / jnp.float32(nx),
    )
    plane = lax.all_gather(plane, AXIS, axis=0, tiled=True)
    return g, plane


def hessian_from_spectra(g, nyquist_plane, x_start, shape):
    """
    The six real Hessian components ``(xx, xy, xz, yy, yz, zz)`` of the
    x-planes ``x_start, x_start + 1, ...`` held in the chunk ``g`` (3, planes, ny, nz//2+1).
    """
    g0, g1, g2 = g[0], g[1], g[2]
    _, ny, nz = shape
    ky_index = jnp.arange(ny)[None, :, None]
    kz_index = jnp.arange(nz // 2 + 1)[None, None, :]
    ky, kz = wavenumbers(ky_index, ny), wavenumbers(kz_index, nz)
    ky1, kz1 = wavenumbers(ky_index, ny, "odd"), wavenumbers(kz_index, nz, "odd")
    ky2, kz2 = wavenumbers(ky_index, ny, "nyquist"), wavenumbers(kz_index, nz, "nyquist")

    # k_x'' part of g_1: the k_x = n_x/2 plane times exp(iπx) = (-1)^x.
    x = x_start + jnp.arange(g1.shape[0])
    sign = jnp.where(x % 2 == 0, jnp.float32(1.0), jnp.float32(-1.0))[:, None, None]
    g1_nyq = sign * nyquist_plane[None]

    def inverse(h):
        return jnp.fft.irfft2(h, s=(ny, nz), axes=(1, 2))

    return (
        inverse(g2),
        inverse(ky1 * g1 + ky2 * g1_nyq),
        inverse(kz1 * g1 + kz2 * g1_nyq),
        inverse((ky * ky) * g0),
        inverse((ky1 * kz1 + ky2 * kz2) * g0),
        inverse((kz * kz) * g0),
    )


def hessian(density, R, mesh, algorithm=NEXUS, floor=None):
    """
    The six Hessian components of the field smoothed at scale ``R``, each
    x-sharded ``(nx, ny, nz)``, stacked as ``(6, nx, ny, nz)``. ``floor`` as
    in ``smooth``.

    For checks on small grids only: it holds all six components at once,
    which ``maximum_signature`` never does.
    """
    check_shape(density.shape, mesh)
    shape = density.shape
    n_dev = mesh.shape[AXIS]
    R = jnp.float32(R)
    floor = resolve_floor(density, mesh, algorithm, floor)

    def local(rho, R, floor):
        if algorithm == NEXUS:
            spec, smooth = local_rfft3(rho), True
        else:
            field = jnp.maximum(rho, floor)
            total = lax.psum(jnp.sum(field), AXIS)
            spec = local_loggauss_spectrum(local_rfft3(jnp.log10(field)), total, R, shape)
            smooth = False
        g, plane = local_hessian_spectra(spec, R, shape, n_dev, n_chunks=1, smooth=smooth)
        x_start = lax.axis_index(AXIS) * (shape[0] // n_dev)
        return jnp.stack(hessian_from_spectra(g, plane, x_start, shape))

    fn = jax.shard_map(
        local, mesh=mesh, in_specs=(REAL_SPEC, P(), P()), out_specs=P(None, AXIS, None, None)
    )
    return jax.jit(fn)(density, R, floor)
