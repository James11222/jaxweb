"""Tagging against PyCosmoMMF, and clump labelling across slab boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from test_pycosmommf_parity import lognormal_cube

import jaxweb
from jaxweb.labels import SUMS, clump_sums
from jaxweb.tagging import CLUMP_STATISTICS, virialized_clumps

pcm = pytest.importorskip("pycosmommf")
skimage_measure = pytest.importorskip("skimage.measure")

RS = [1.0, 2.0**0.5, 2.0, 2.0**1.5]
# PyCosmoMMF's cluster-signature grid, which jaxweb's default refines and extends.
PCM_CLUSTER_THRESHOLDS = jaxweb.tagging.cluster_signature_thresholds(smax=30, n=10)


def reference_clump_means(mask, delta):
    labels, n = skimage_measure.label(mask, connectivity=3, return_num=True)
    sums = np.bincount(labels.ravel(), weights=delta.ravel(), minlength=n + 1)[1:]
    counts = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    return np.sort(sums / counts)


def clump_means(mask, rho, mesh):
    sums = clump_sums(mask, rho, mesh)
    return np.sort(sums["delta"] / sums["count"])


@pytest.mark.parametrize("fill", [0.05, 0.2, 0.3])
def test_clumps_join_across_slabs(mesh, fill):
    rng = np.random.default_rng(int(fill * 100))
    mask = rng.random((32, 24, 20)) < fill  # many clumps cross the 3 slab boundaries
    delta = rng.standard_normal(mask.shape).astype(np.float32)
    got = clump_means(mask, delta + np.float32(1.0), mesh)
    want = reference_clump_means(mask, delta)
    assert len(got) == len(want)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_labelling_in_pieces_within_a_slab(mesh, monkeypatch):
    monkeypatch.setattr(jaxweb.labels, "LABEL_ELEMENTS", 2 * 24 * 20)  # 2-plane pieces
    rng = np.random.default_rng(5)
    mask = rng.random((32, 24, 20)) < 0.25
    delta = rng.standard_normal(mask.shape).astype(np.float32)
    got = clump_means(mask, delta + np.float32(1.0), mesh)
    np.testing.assert_allclose(got, reference_clump_means(mask, delta), rtol=1e-5, atol=1e-5)


def test_clump_touching_only_diagonally_across_a_boundary(mesh):
    mask = np.zeros((16, 8, 8), dtype=bool)
    mask[3, 2, 2] = True  # last plane of slab 0
    mask[4, 3, 3] = True  # first plane of slab 1, diagonal neighbour
    mask[4, 6, 6] = True  # separate clump
    delta = np.zeros(mask.shape, dtype=np.float32)
    delta[3, 2, 2], delta[4, 3, 3], delta[4, 6, 6] = 1.0, 3.0, 5.0
    rho = delta + np.float32(1.0)
    np.testing.assert_allclose(clump_means(mask, rho, mesh), [2.0, 5.0])


def test_every_sum_joins_across_slabs(mesh):
    rng = np.random.default_rng(11)
    mask = rng.random((32, 24, 20)) < 0.1
    # Integer ρ, so every float32 sum is exact and clumps match row by row.
    rho = rng.integers(0, 40, mask.shape).astype(np.float32)
    delta = rho - np.float32(1.0)
    threshold = 19.5
    got = clump_sums(mask, rho, mesh, SUMS, threshold)
    labels, n = skimage_measure.label(mask, connectivity=3, return_num=True)
    weights = {
        "count": np.ones_like(rho),
        "delta": delta,
        "mass": rho,
        "mass_delta": rho * delta,
        "above": delta > threshold,
    }
    want = {
        name: np.bincount(labels.ravel(), weights=w.ravel(), minlength=n + 1)[1:]
        for name, w in weights.items()
    }
    assert got["count"].dtype == got["above"].dtype == np.int64
    assert got["mass_delta"].dtype == np.float32

    def rows(sums):
        table = np.stack([sums[name] for name in SUMS], axis=1).astype(np.float64)
        return table[np.lexsort(table.T[::-1])]

    np.testing.assert_array_equal(rows(got), rows(want))


@pytest.mark.parametrize("statistic", list(CLUMP_STATISTICS))
def test_virialized_clumps_match_direct_statistics(mesh, statistic):
    rng = np.random.default_rng(13)
    mask = rng.random((32, 24, 20)) < 0.1
    rho = rng.lognormal(0.0, 2.0, mask.shape).astype(np.float32)
    delta = rho - np.float32(1.0)
    threshold = 5.0
    sums = clump_sums(mask, rho, mesh, CLUMP_STATISTICS[statistic], threshold)
    got = virialized_clumps(sums, statistic, threshold)
    labels, n = skimage_measure.label(mask, connectivity=3, return_num=True)
    direct = {
        "mean": np.mean,
        "mass_weighted": lambda d: np.sum((d + 1) * d) / np.sum(d + 1),
        "median": lambda d: np.sort(d)[(len(d) - 1) // 2],
        "max": np.max,
    }[statistic]
    want = np.array([direct(delta[labels == i]) for i in range(1, n + 1)]) > threshold
    assert len(got) == n
    assert 0 < np.count_nonzero(want) < n
    assert np.count_nonzero(got) == np.count_nonzero(want)


def test_clump_statistics_are_ordered(mesh):
    # Per clump, mean ≤ mass-weighted mean ≤ max and median ≤ max, so each
    # virialized set contains the one before.
    rng = np.random.default_rng(17)
    mask = rng.random((32, 24, 20)) < 0.1
    rho = rng.lognormal(0.0, 2.0, mask.shape).astype(np.float32)
    sums = clump_sums(mask, rho, mesh, SUMS, 5.0)
    v = {s: virialized_clumps(sums, s, 5.0) for s in CLUMP_STATISTICS}
    assert not np.any(v["mean"] & ~v["mass_weighted"])
    assert not np.any(v["mass_weighted"] & ~v["max"])
    assert not np.any(v["median"] & ~v["max"])


@pytest.fixture(scope="module")
def tagged_cube():
    rho = lognormal_cube(64, seed=7)
    return rho, pcm.maximum_signature(RS, rho, algorithm="NEXUSPLUS")


@pytest.mark.parametrize("overdensity_threshold", [10, 40])
def test_tagging_matches_pycosmommf(mesh, tagged_cube, overdensity_threshold, capsys):
    rho, sigs = tagged_cube
    want, want_summary = pcm.calc_structure_bools(
        rho, sigs, True, overdensity_threshold=overdensity_threshold
    )
    capsys.readouterr()
    got, summary = jaxweb.tag_structures(
        rho,
        np.moveaxis(sigs, -1, 0),
        overdensity_threshold=overdensity_threshold,
        cluster_thresholds=PCM_CLUSTER_THRESHOLDS,
        clump_statistic="mean",
        mesh=mesh,
    )
    got = np.asarray(got)
    assert got.dtype == np.uint8
    assert summary["S_fil_thresh"] == want_summary["S_fil_thresh"]
    assert summary["S_wall_thresh"] == want_summary["S_wall_thresh"]
    np.testing.assert_array_equal(summary["f_vir_clus"], want_summary["f_vir_clus"])
    # The curves differ by float32 summation order (tree sums vs NumPy's
    # pairwise sums), which dM² amplifies; the thresholds above match exactly.
    np.testing.assert_allclose(summary["dM2_fil"], want_summary["dM2_fil"], rtol=1e-3)
    np.testing.assert_allclose(summary["dM2_wall"], want_summary["dM2_wall"], rtol=1e-3)
    np.testing.assert_array_equal(got, want)
    assert np.any(got == jaxweb.CLUSTER)
    np.testing.assert_allclose(summary["volume_fractions"], want_summary["volume_fractions"])
    np.testing.assert_allclose(summary["mass_fractions"], want_summary["mass_fractions"], rtol=1e-5)


def test_clump_statistic_shifts_the_virialized_fraction(mesh, tagged_cube):
    rho, sigs = tagged_cube
    f_vir = {}
    for statistic in CLUMP_STATISTICS:
        _, summary = jaxweb.tag_structures(
            rho,
            np.moveaxis(sigs, -1, 0),
            overdensity_threshold=40,
            clump_statistic=statistic,
            mesh=mesh,
        )
        assert summary["clump_statistic"] == statistic
        f_vir[statistic] = summary["f_vir_clus"]
    resolved = np.isfinite(f_vir["mean"])
    assert np.all(f_vir["mass_weighted"][resolved] >= f_vir["mean"][resolved])
    assert np.all(f_vir["max"][resolved] >= f_vir["mass_weighted"][resolved])
    assert np.any(f_vir["max"][resolved] > f_vir["mean"][resolved])


def test_unknown_clump_statistic_raises(mesh, tagged_cube):
    rho, sigs = tagged_cube
    with pytest.raises(ValueError, match="clump_statistic"):
        jaxweb.tag_structures(rho, np.moveaxis(sigs, -1, 0), clump_statistic="mode", mesh=mesh)


def test_unresolved_clusters_warn_and_leave_no_clusters(mesh, tagged_cube, capsys):
    rho, sigs = tagged_cube
    got, summary = jaxweb.tag_structures(
        rho, np.moveaxis(sigs, -1, 0), overdensity_threshold=1e6, mesh=mesh
    )
    assert "WARNING" in capsys.readouterr().out
    assert summary["S_clus_thresh"] is None
    assert not np.any(np.asarray(got) == jaxweb.CLUSTER)


def test_overdensity_input_raises(mesh, tagged_cube):
    rho, sigs = tagged_cube
    with pytest.raises(ValueError, match="δ"):
        jaxweb.tag_structures(rho - 1.0, np.moveaxis(sigs, -1, 0), mesh=mesh)


def test_end_to_end_matches_pycosmommf(mesh, capsys):
    rho = lognormal_cube(64, seed=7)
    want, want_summary = pcm.calc_structure_bools(
        rho, pcm.maximum_signature(RS, rho), True, overdensity_threshold=20
    )
    capsys.readouterr()
    got, summary = jaxweb.tag_structures(
        rho,
        jaxweb.maximum_signature(rho, RS, mesh=mesh, floor=rho.min()),
        overdensity_threshold=20,
        cluster_thresholds=PCM_CLUSTER_THRESHOLDS,
        clump_statistic="mean",
        mesh=mesh,
    )
    assert summary["S_fil_thresh"] == want_summary["S_fil_thresh"]
    assert summary["S_wall_thresh"] == want_summary["S_wall_thresh"]
    # Signatures differ at float32 FFT precision, which flips the odd voxel
    # that sits right at a threshold.
    assert np.mean(np.asarray(got) != want) < 1e-4
