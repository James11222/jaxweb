"""Multi-scale NEXUS / NEXUS+ maximum signatures on a sharded density cube."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec as P

from .eigen import eigvalsh_3x3_sym, signatures_from_eigenvalues
from .fft import largest_divisor_at_most, local_rfft3
from .hessian import hessian_from_spectra, local_hessian_spectra
from .mesh import (
    AXIS,
    CHUNK_ELEMENTS,
    KSPACE_SPEC,
    REAL_SPEC,
    SIG_SPEC,
    as_sharded,
    check_shape,
    kspace_sharding,
    real_sharding,
    sig_sharding,
)
from .smoothing import ALGORITHMS, NEXUS, NEXUSPLUS, local_loggauss_spectrum


def _chunking(shape, n_dev, chunk_elements):
    """``(ky_chunks, x_chunks)``: loop counts for the transpose and signature stages."""
    nx, ny, nz = shape
    nzh = nz // 2 + 1
    ny_local, nx_local = ny // n_dev, nx // n_dev
    ky_per_chunk = largest_divisor_at_most(ny_local, chunk_elements // (nx * nzh))
    x_per_chunk = largest_divisor_at_most(nx_local, chunk_elements // (ny * nz))
    return ny_local // ky_per_chunk, nx_local // x_per_chunk


def _local_update_signatures(sigmax, g, nyquist_plane, shape, x_chunks):
    """Running max of the signatures, a few x-planes at a time."""
    nx_local = sigmax.shape[1]
    planes = nx_local // x_chunks
    x_offset = lax.axis_index(AXIS) * nx_local

    def body(c, sigmax):
        start = c * planes
        chunk = lax.dynamic_slice_in_dim(g, start, planes, axis=1)
        hess = hessian_from_spectra(chunk, nyquist_plane, x_offset + start, shape)
        eigs = eigvalsh_3x3_sym(*hess)
        sigs = signatures_from_eigenvalues(*eigs)
        current = lax.dynamic_slice_in_dim(sigmax, start, planes, axis=1)
        return lax.dynamic_update_slice_in_dim(sigmax, jnp.maximum(current, sigs), start, axis=1)

    return lax.fori_loop(0, x_chunks, body, sigmax)


def _make_signature_step(mesh, shape, chunk_elements, smooth):
    """
    Jitted, sharded ``(sigmax, spec, R) -> sigmax``: Hessian of ``spec`` at
    scale R, signatures, running max. With ``smooth=True`` (NEXUS) ``spec``
    is the unsmoothed spectrum and the Gaussian is applied on the fly.
    ``sigmax`` is donated and updated in place.
    """
    n_dev = mesh.shape[AXIS]
    ky_chunks, x_chunks = _chunking(shape, n_dev, chunk_elements)

    def local(sigmax, spec, R):
        g, plane = local_hessian_spectra(spec, R, shape, n_dev, ky_chunks, smooth=smooth)
        return _local_update_signatures(sigmax, g, plane, shape, x_chunks)

    fn = jax.shard_map(local, mesh=mesh, in_specs=(SIG_SPEC, KSPACE_SPEC, P()), out_specs=SIG_SPEC)
    return jax.jit(fn, donate_argnums=0)


def _make_loggauss_step(mesh, shape):
    """Jitted ``(log_spec, total, R) -> spec``: the NEXUS+ smoothed field's spectrum."""
    fn = jax.shard_map(
        lambda base, total, R: local_loggauss_spectrum(base, total, R, shape),
        mesh=mesh,
        in_specs=(KSPACE_SPEC, P(), P()),
        out_specs=KSPACE_SPEC,
    )
    return jax.jit(fn)


def _slab_bytes(shape, mesh):
    return 4 * shape[0] * shape[1] * shape[2] // mesh.shape[AXIS]


def _should_offload(shape, mesh):
    """
    Park the NEXUS+ log-spectrum in host memory when ~8.3 field slabs per device
    (the on-device peak, see ``benchmarks/memory_estimate.py``) would not fit
    in 90% of the device memory. Never on devices that report no limit (CPU).
    """
    stats = mesh.devices.flat[0].memory_stats() or {}
    limit = stats.get("bytes_limit")
    return bool(limit) and 8.3 * _slab_bytes(shape, mesh) > 0.9 * limit


def _make_base(mesh, shape, algorithm, eps):
    """k-space of ρ (NEXUS) or log10 ρ (NEXUS+), and the total mass Σρ."""

    def local(rho):
        field = rho + jnp.float32(eps)
        total = lax.psum(jnp.sum(field), AXIS)
        if algorithm == NEXUSPLUS:
            field = jnp.log10(field)
        return local_rfft3(field), total

    fn = jax.shard_map(local, mesh=mesh, in_specs=REAL_SPEC, out_specs=(KSPACE_SPEC, P()))
    return jax.jit(fn)


def maximum_signature(
    density,
    Rs,
    algorithm=NEXUSPLUS,
    mesh=None,
    eps=1e-16,
    chunk_elements=CHUNK_ELEMENTS,
    offload=None,
):
    """
    Maximum cluster, filament and wall signatures over the smoothing scales ``Rs``.

    Args:
        density: 3D δ+1 = ρ/⟨ρ⟩ field of a periodic box, a NumPy array or an
            x-sharded JAX array (see ``jaxweb.load_density``). Converted to
            float32.
        Rs: smoothing scales in voxels, e.g. ``[2 ** (n / 2) for n in range(9)]``.
        algorithm: ``"NEXUS"`` (Gaussian smoothing of ρ) or ``"NEXUSPLUS"``
            (Gaussian smoothing of log10 ρ, which needs ρ > 0).
        mesh: the device mesh; default ``jaxweb.make_mesh()`` over every device.
        eps: added to the density before use, and the floor of the result,
            as in PyCosmoMMF.
        chunk_elements: elements per chunk in the chunked loops; lower it to
            trade speed for memory.
        offload: NEXUS+ only. ``True`` keeps the spectrum of log10 ρ in pinned
            host memory and copies it to the devices once per scale, saving
            one field slab of device memory per device for a host-device copy
            per scale. ``None`` (default) decides from the device memory.

    Returns:
        float32 JAX array ``(3, nx, ny, nz)``, sharded along x: the cluster,
        filament and wall signatures, in that order.
    """
    from .mesh import make_mesh

    if algorithm not in ALGORITHMS:
        msg = f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}"
        raise ValueError(msg)
    mesh = make_mesh() if mesh is None else mesh
    check_shape(density.shape, mesh)
    shape = tuple(density.shape)

    density = as_sharded(density, real_sharding(mesh), jnp.float32)
    if algorithm == NEXUSPLUS:
        n_negative = int(jnp.sum(density < 0))
        if n_negative:
            msg = (
                f"NEXUSPLUS uses log10(density) and needs a positive field; got "
                f"{n_negative} negative voxels. Pass δ+1 = ρ/⟨ρ⟩, not δ."
            )
            raise ValueError(msg)

    base, total = _make_base(mesh, shape, algorithm, eps)(density)
    del density

    sigmax = jax.jit(
        lambda: jnp.full((3, *shape), eps, dtype=jnp.float32),
        out_shardings=sig_sharding(mesh),
    )()

    if algorithm == NEXUS:
        step = _make_signature_step(mesh, shape, chunk_elements, smooth=True)
        for R in Rs:
            sigmax = step(sigmax, base, jnp.float32(R))
        return sigmax

    smooth_step = _make_loggauss_step(mesh, shape)
    step = _make_signature_step(mesh, shape, chunk_elements, smooth=False)
    if offload is None:
        offload = _should_offload(shape, mesh)
    if offload:
        base = jax.device_put(base, kspace_sharding(mesh).with_memory_kind("pinned_host"))
    for R in Rs:
        R = jnp.float32(R)
        spec = smooth_step(
            jax.device_put(base, kspace_sharding(mesh)) if offload else base, total, R
        )
        sigmax = step(sigmax, spec, R)
        del spec
    return sigmax


def to_pycosmommf_layout(sigmax):
    """``(3, nx, ny, nz)`` signatures as a NumPy ``(nx, ny, nz, 3)`` array (small grids only)."""
    return np.moveaxis(np.asarray(sigmax), 0, -1)


__all__ = ["NEXUS", "NEXUSPLUS", "maximum_signature", "to_pycosmommf_layout"]
