"""Cluster / filament / wall / void tagging from maximum signatures.

Follows PyCosmoMMF's ``calc_structure_bools`` step for step, on x-sharded
arrays: the same thresholds, the same selection rules and the same uint8
``structure_filter`` (1 cluster, 2 filament, 3 wall, 4 void).

1. Clusters: for each cluster-signature threshold, label the clumps above it
   and take the fraction that are virialized; keep the threshold whose
   fraction is closest to 0.5. By default a clump is virialized when its
   largest δ exceeds ``overdensity_threshold``; PyCosmoMMF uses the mean δ
   (``clump_statistic="mean"``; see ``virialized_clumps``).
2. Filaments, then walls: the enclosed mass M(𝒮) over a log10 𝒮 grid, and
   the threshold at the peak of |dM²/dlog 𝒮|.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec as P

from .fft import largest_divisor_at_most
from .labels import clump_sums
from .mesh import (
    AXIS,
    CHUNK_ELEMENTS,
    REAL_SPEC,
    SIG_SPEC,
    as_sharded,
    check_shape,
    make_mesh,
    real_sharding,
    sig_sharding,
)

CLUSTER, FILAMENT, WALL, VOID = 1, 2, 3, 4

# Per-clump δ statistics a clump's virialization can be judged by, and the
# ``labels.clump_sums`` each needs. jaxweb uses the max, PyCosmoMMF the mean.
CLUMP_STATISTICS = {
    "mean": ("count", "delta"),
    "mass_weighted": ("mass", "mass_delta"),
    "median": ("count", "above"),
    "max": ("above",),
}


# -- Helpers shared with PyCosmoMMF (same rules, same float32 values) --------


def cluster_signature_thresholds(smax=60.0, n=20):
    """
    The float32 cluster-signature thresholds scanned for clusters: ``n``
    log-spaced values from 0.1 to ``smax``. PyCosmoMMF uses ``smax=30, n=10``;
    the default is twice as fine and reaches twice as high.
    """
    return np.geomspace(0.1, smax, n, dtype=np.float32)


def virialized_fraction(n_virialized, n_clumps):
    """Fraction of clumps that are virialized, as float32; NaN if there are no clumps."""
    if n_clumps == 0:
        return np.float32(np.nan)
    return np.float32(n_virialized) / np.float32(n_clumps)


def select_cluster_threshold(signature_thresholds, virialized_fractions):
    """The threshold whose virialized fraction is closest to 0.5; ``None`` if none is virialized."""
    if not np.any(virialized_fractions > 0):
        return None
    distance = np.abs(virialized_fractions - np.float32(0.5))
    return signature_thresholds[np.nanargmin(distance)]


def log10S_grid(Smin, Smax):
    """float32 log10(𝒮) grid: the points of ``np.arange(Smin, Smax + 0.1, 0.1)``."""
    n = int(np.ceil(round((Smax + 0.1 - Smin) / 0.1, 6)))
    return np.float32(Smin) + np.float32(0.1) * np.arange(n, dtype=np.float32)


def mass_change_curve(log10S, M):
    """``|dM²/dlog(𝒮)|`` at the midpoints of ``log10S``, and 𝒮 at those midpoints."""
    dM2 = np.abs(np.diff(M**2) / np.diff(log10S))
    mid = (log10S[:-1] + log10S[1:]) * np.float32(0.5)
    return np.power(np.float32(10.0), mid), dM2


def warn_unresolved_clusters(overdensity_threshold, clump_statistic="mean"):
    print(
        f"WARNING: no clump has a {clump_statistic} overdensity δ above "
        f"{overdensity_threshold} at any cluster-signature threshold. The "
        "resolution of the grid is not sufficiently high to resolve virialized "
        "clusters/nodes of the cosmic web, so the cluster filter is empty."
    )


def _check_clump_statistic(clump_statistic):
    if clump_statistic not in CLUMP_STATISTICS:
        msg = f"clump_statistic must be one of {list(CLUMP_STATISTICS)}, not {clump_statistic!r}"
        raise ValueError(msg)


def virialized_clumps(sums, clump_statistic, overdensity_threshold):
    """
    Boolean per clump: whether its δ statistic exceeds ``overdensity_threshold``,
    from its ``labels.clump_sums`` (``CLUMP_STATISTICS`` lists the sums each
    statistic needs, with ``"above"`` counted above ``overdensity_threshold``).

    - ``"mean"``: Σδ / N, as PyCosmoMMF.
    - ``"mass_weighted"``: Σρδ / Σρ, never below the mean.
    - ``"median"``: the lower median (element ``(N - 1) // 2`` of the sorted δ),
      which exceeds the threshold exactly when more than half the voxels do.
      Usually below the mean in a centrally concentrated clump.
    - ``"max"``: the largest δ, which exceeds it when at least one voxel does.
    """
    _check_clump_statistic(clump_statistic)
    t = np.float32(overdensity_threshold)
    if clump_statistic == "mean":
        return sums["delta"] / sums["count"].astype(np.float32) > t
    if clump_statistic == "mass_weighted":
        return sums["mass_delta"] / sums["mass"] > t
    if clump_statistic == "median":
        return 2 * sums["above"] > sums["count"]
    return sums["above"] > 0  # max


# -- Device computations ------------------------------------------------------


def _make_masked_masses(mesh, component):
    """
    Jitted float32 Σ ρ over ``valid & (sigmax[component] > t)`` for each
    threshold ``t``, chunked over x. Takes the whole signature array, so no
    copy of one component is made.
    """

    def local(sigmax, rho, valid, thresholds):
        sig = sigmax[component]
        nx, ny, nz = sig.shape
        planes = largest_divisor_at_most(nx, CHUNK_ELEMENTS // (ny * nz))

        def mass_above(t):
            def body(c, total):
                s, r, v = (
                    lax.dynamic_slice_in_dim(a, c * planes, planes, axis=0)
                    for a in (sig, rho, valid)
                )
                return total + jnp.sum(jnp.where(v & (s > t), r, 0.0))

            zero = lax.pcast(jnp.float32(0.0), (AXIS,), to="varying")
            return lax.fori_loop(0, nx // planes, body, zero)

        return lax.psum(lax.map(mass_above, thresholds), AXIS)

    fn = jax.shard_map(
        local, mesh=mesh, in_specs=(SIG_SPEC, REAL_SPEC, REAL_SPEC, P()), out_specs=P()
    )
    return jax.jit(fn)


@jax.jit
def _filament_valid(clusbool):
    return ~clusbool


@jax.jit
def _wall_valid(clusbool, sigmax, filament_thresh):
    return ~clusbool & (sigmax[1] < filament_thresh)


@jax.jit
def _cluster_mask(sigmax, S_th):
    return sigmax[0] > S_th


@jax.jit
def _structure_filter(clusbool, sigmax, filament_thresh, wall_thresh):
    filbool = (sigmax[1] > filament_thresh) & ~clusbool
    wallbool = (sigmax[2] > wall_thresh) & ~(filbool | clusbool)
    out = jnp.where(wallbool, jnp.uint8(WALL), jnp.uint8(VOID))
    out = jnp.where(filbool, jnp.uint8(FILAMENT), out)
    return jnp.where(clusbool, jnp.uint8(CLUSTER), out)


def _make_fractions(mesh):
    """
    Jitted per-type voxel counts and float32 masses, one pass, chunked over x.
    Counts come back per chunk (int32 cannot hold a 2048³ total) and are
    added on the host.
    """

    def local(structure, rho):
        nx, ny, nz = structure.shape
        planes = largest_divisor_at_most(nx, CHUNK_ELEMENTS // (ny * nz))
        kinds = (CLUSTER, FILAMENT, WALL, VOID)

        def body(c, mass):
            st, r = (
                lax.dynamic_slice_in_dim(a, c * planes, planes, axis=0) for a in (structure, rho)
            )
            counts = jnp.stack([jnp.sum(st == k, dtype=jnp.int32) for k in kinds])
            mass = mass + jnp.stack([jnp.sum(jnp.where(st == k, r, 0.0)) for k in kinds])
            return mass, counts

        mass = lax.pcast(jnp.zeros(4, jnp.float32), (AXIS,), to="varying")
        mass, counts = lax.scan(lambda m, c: body(c, m), mass, jnp.arange(nx // planes))
        return counts, lax.psum(mass, AXIS)

    fn = jax.shard_map(local, mesh=mesh, in_specs=(REAL_SPEC, REAL_SPEC), out_specs=(P(AXIS), P()))
    return jax.jit(fn)


def _mass_threshold(sigmax, component, rho, valid, Smin, Smax, mesh):
    log10S = log10S_grid(Smin, Smax)
    thresholds = np.power(np.float32(10.0), log10S)
    masses = _make_masked_masses(mesh, component)(sigmax, rho, valid, jnp.asarray(thresholds))
    M = np.asarray(masses, dtype=np.float32)
    S, dM2 = mass_change_curve(log10S, M)
    return S[np.argmax(dM2)], S, dM2


def _cluster_bool(rho, sigmax, overdensity_threshold, thresholds, clump_statistic, mesh):
    names = CLUMP_STATISTICS[clump_statistic]
    fractions = np.zeros_like(thresholds)
    for i, S_th in enumerate(thresholds):
        sums = clump_sums(_cluster_mask(sigmax, S_th), rho, mesh, names, overdensity_threshold)
        virialized = virialized_clumps(sums, clump_statistic, overdensity_threshold)
        fractions[i] = virialized_fraction(np.count_nonzero(virialized), len(virialized))
    S_th = select_cluster_threshold(thresholds, fractions)
    if S_th is None:
        warn_unresolved_clusters(overdensity_threshold, clump_statistic)
        clusbool = _cluster_mask(sigmax, np.float32(np.inf))
    else:
        clusbool = _cluster_mask(sigmax, S_th)
    return clusbool, S_th, thresholds, fractions


def tag_structures(
    density,
    sigmax,
    overdensity_threshold=370,
    Smin=-3,
    Smax=2,
    clusbool=None,
    cluster_thresholds=None,
    clump_statistic="max",
    mesh=None,
    verbose=False,
):
    """
    Tag every voxel as cluster, filament, wall or void.

    Args:
        density: δ+1 = ρ/⟨ρ⟩, NumPy or x-sharded JAX ``(nx, ny, nz)``.
        sigmax: signatures from ``maximum_signature``, ``(3, nx, ny, nz)``.
        overdensity_threshold: the δ a clump's ``clump_statistic`` must exceed
            for it to count as virialized (commonly 370, 200 or 500).
        Smin, Smax: range of the log10 𝒮 grid for the filament and wall
            mass curves.
        clusbool: an optional boolean cluster mask to use instead of step 1.
        cluster_thresholds: the cluster-signature thresholds scanned in
            step 1; default ``cluster_signature_thresholds()``.
        clump_statistic: the per-clump δ statistic compared with
            ``overdensity_threshold`` in step 1, one of ``CLUMP_STATISTICS``:
            ``"max"`` (the default), ``"mean"`` (PyCosmoMMF's),
            ``"mass_weighted"`` or ``"median"``; see ``virialized_clumps``.
        mesh: the device mesh; default ``jaxweb.make_mesh()``.
        verbose: print PyCosmoMMF's report.

    Returns:
        ``(structure_filter, summary)``: an x-sharded uint8 array
        (1 cluster, 2 filament, 3 wall, 4 void) and a dict with the
        thresholds, curves and volume / mass fractions, using the keys of
        PyCosmoMMF's ``summary_data``, plus ``"clump_statistic"`` and
        ``"seconds"``: the wall-clock time of each stage (clusters, filaments,
        walls, filter).
    """
    _check_clump_statistic(clump_statistic)
    mesh = make_mesh() if mesh is None else mesh
    check_shape(density.shape, mesh)
    rho = as_sharded(density, real_sharding(mesh), jnp.float32)
    sigmax = as_sharded(sigmax, sig_sharding(mesh), jnp.float32)

    n_cells = np.float32(np.prod(rho.shape))
    if np.isclose(np.abs(float(jnp.mean(rho))), 0, atol=1e-3):
        msg = (
            "make sure that you are not inputing the overdensity field δ = ρ/<ρ> - 1, "
            "but rather δ + 1 = ρ/<ρ>."
        )
        raise ValueError(msg)

    seconds = {}
    t = time.perf_counter()
    if clusbool is None:
        if cluster_thresholds is None:
            cluster_thresholds = cluster_signature_thresholds()
        clusbool, cluster_thresh, S_clus, f_vir_clus = _cluster_bool(
            rho,
            sigmax,
            overdensity_threshold,
            np.float32(cluster_thresholds),
            clump_statistic,
            mesh,
        )
    else:
        clusbool = as_sharded(clusbool, real_sharding(mesh), jnp.bool_)
        cluster_thresh = S_clus = f_vir_clus = None
    clusbool.block_until_ready()
    seconds["clusters"] = time.perf_counter() - t

    t = time.perf_counter()
    filament_thresh, S_fil, dM2_fil = _mass_threshold(
        sigmax, 1, rho, _filament_valid(clusbool), Smin, Smax, mesh
    )
    seconds["filaments"] = time.perf_counter() - t
    t = time.perf_counter()
    wall_thresh, S_wall, dM2_wall = _mass_threshold(
        sigmax, 2, rho, _wall_valid(clusbool, sigmax, filament_thresh), Smin, Smax, mesh
    )
    seconds["walls"] = time.perf_counter() - t

    t = time.perf_counter()
    structure = _structure_filter(clusbool, sigmax, filament_thresh, wall_thresh)
    volume, mass = (np.asarray(a) for a in _make_fractions(mesh)(structure, rho))
    seconds["filter"] = time.perf_counter() - t
    volume = volume.sum(axis=0, dtype=np.int64)
    total_mass = np.float32(mass.sum())
    summary = {
        "clump_statistic": clump_statistic,
        "S_clus_thresh": cluster_thresh,
        "S_clus": S_clus,
        "f_vir_clus": f_vir_clus,
        "S_fil": S_fil,
        "dM2_fil": dM2_fil,
        "S_fil_thresh": filament_thresh,
        "S_wall": S_wall,
        "dM2_wall": dM2_wall,
        "S_wall_thresh": wall_thresh,
        "volume_fractions": [np.float32(v) / n_cells for v in volume],
        "mass_fractions": [np.float32(m) / total_mass for m in mass],
        "seconds": seconds,
    }
    if verbose:
        _report(summary)
    return structure, summary


def _report(summary):
    print(f"Clump statistic: {summary['clump_statistic']}")
    names = ("Clusters", "Filaments", "Walls", "Voids")
    thresholds = (
        summary["S_clus_thresh"],
        summary["S_fil_thresh"],
        summary["S_wall_thresh"],
        None,
    )
    for name, thresh, vol, mass in zip(
        names, thresholds, summary["volume_fractions"], summary["mass_fractions"], strict=True
    ):
        print(f"{name}: signature threshold {thresh}, volume fraction {vol}, mass fraction {mass}")
