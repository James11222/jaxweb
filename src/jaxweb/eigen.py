"""Eigenvalues of the Hessian and the NEXUS morphology signatures, elementwise."""

from __future__ import annotations

import math

import jax.numpy as jnp


def eigvalsh_3x3_sym(a, b, c, d, e, f):
    """
    Sorted eigenvalues ``(e1 <= e2 <= e3)`` of the symmetric matrices

        [[a, b, c],
         [b, d, e],
         [c, e, f]]

    given each entry as an array. Closed-form trigonometric solution of
    Smith (1961), Communications of the ACM 4, 168, in float32. It is the
    same solver, with the same max-abs rescaling, as PyCosmoMMF's
    ``_eigvalsh_3x3_sym``, so the two codes agree to float32 rounding.
    """
    scale = jnp.maximum(
        jnp.maximum(jnp.maximum(jnp.abs(a), jnp.abs(d)), jnp.maximum(jnp.abs(f), jnp.abs(b))),
        jnp.maximum(jnp.abs(c), jnp.abs(e)),
    )
    all_zero = scale == 0.0
    inv_scale = 1.0 / jnp.where(all_zero, 1.0, scale)

    a_s, b_s, c_s = a * inv_scale, b * inv_scale, c * inv_scale
    d_s, e_s, f_s = d * inv_scale, e * inv_scale, f * inv_scale

    p1 = b_s * b_s + c_s * c_s + e_s * e_s
    q = (a_s + d_s + f_s) / 3.0
    a_q, d_q, f_q = a_s - q, d_s - q, f_s - q
    p = jnp.sqrt((a_q * a_q + d_q * d_q + f_q * f_q + 2.0 * p1) / 6.0)

    det_shifted = (
        a_q * (d_q * f_q - e_s * e_s)
        - b_s * (b_s * f_q - e_s * c_s)
        + c_s * (b_s * e_s - d_q * c_s)
    )

    degenerate = p == 0.0
    safe_p = jnp.where(degenerate, 1.0, p)
    r = jnp.clip(det_shifted / (2.0 * safe_p**3), -1.0, 1.0)

    phi = jnp.arccos(r) / 3.0
    eig_max = q + 2.0 * p * jnp.cos(phi)
    eig_min = q + 2.0 * p * jnp.cos(phi + jnp.float32(2.0 * math.pi / 3.0))
    # Trace conservation gives the middle eigenvalue without another trig call.
    eig_mid = 3.0 * q - eig_max - eig_min

    def finish(eig):
        eig = jnp.where(degenerate, q, eig)
        return jnp.where(all_zero, 0.0, eig * scale).astype(jnp.float32)

    return finish(eig_min), finish(eig_mid), finish(eig_max)


def _pos_part(x):
    """x · Heaviside(x)."""
    return jnp.maximum(x, 0.0)


def signatures_from_eigenvalues(e1, e2, e3):
    """
    Cluster, filament and wall signatures from sorted eigenvalues
    ``e1 <= e2 <= e3`` (Cautun et al. 2013, eq. 7), stacked as ``(3, ...)``.

    Same formulas as PyCosmoMMF's ``signatures_from_hessian``; voxels with
    ``e1 == 0`` get zero for all three.
    """
    zero_e1 = e1 == 0.0
    safe_e1 = jnp.where(zero_e1, 1.0, e1)
    r2 = jnp.abs(e2 / safe_e1)
    r3 = jnp.abs(e3 / safe_e1)

    neg1, neg2, neg3 = e1 < 0.0, e2 < 0.0, e3 < 0.0

    cluster = jnp.where(neg1 & neg2 & neg3, r3 * jnp.abs(e3), 0.0)
    filament = jnp.where(neg1 & neg2, r2 * _pos_part(1.0 - r3) * jnp.abs(e2), 0.0)
    wall = jnp.where(neg1, _pos_part(1.0 - r2) * _pos_part(1.0 - r3) * jnp.abs(e1), 0.0)

    sigs = jnp.stack([cluster, filament, wall])
    return jnp.where(zero_e1, 0.0, sigs).astype(jnp.float32)
