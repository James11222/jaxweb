"""Tagging against PyCosmoMMF, and clump labelling across slab boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from test_pycosmommf_parity import lognormal_cube

import jaxweb
from jaxweb.labels import clump_mean_overdensity

pcm = pytest.importorskip("pycosmommf")
skimage_measure = pytest.importorskip("skimage.measure")

RS = [1.0, 2.0**0.5, 2.0, 2.0**1.5]


def reference_clump_means(mask, delta):
    labels, n = skimage_measure.label(mask, connectivity=3, return_num=True)
    sums = np.bincount(labels.ravel(), weights=delta.ravel(), minlength=n + 1)[1:]
    counts = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    return np.sort(sums / counts)


@pytest.mark.parametrize("fill", [0.05, 0.2, 0.3])
def test_clumps_join_across_slabs(mesh, fill):
    rng = np.random.default_rng(int(fill * 100))
    mask = rng.random((32, 24, 20)) < fill  # many clumps cross the 3 slab boundaries
    delta = rng.standard_normal(mask.shape).astype(np.float32)
    got = np.sort(clump_mean_overdensity(mask, delta + np.float32(1.0), mesh))
    want = reference_clump_means(mask, delta)
    assert len(got) == len(want)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_labelling_in_pieces_within_a_slab(mesh, monkeypatch):
    monkeypatch.setattr(jaxweb.labels, "LABEL_ELEMENTS", 2 * 24 * 20)  # 2-plane pieces
    rng = np.random.default_rng(5)
    mask = rng.random((32, 24, 20)) < 0.25
    delta = rng.standard_normal(mask.shape).astype(np.float32)
    got = np.sort(clump_mean_overdensity(mask, delta + np.float32(1.0), mesh))
    np.testing.assert_allclose(got, reference_clump_means(mask, delta), rtol=1e-5, atol=1e-5)


def test_clump_touching_only_diagonally_across_a_boundary(mesh):
    mask = np.zeros((16, 8, 8), dtype=bool)
    mask[3, 2, 2] = True  # last plane of slab 0
    mask[4, 3, 3] = True  # first plane of slab 1, diagonal neighbour
    mask[4, 6, 6] = True  # separate clump
    delta = np.zeros(mask.shape, dtype=np.float32)
    delta[3, 2, 2], delta[4, 3, 3], delta[4, 6, 6] = 1.0, 3.0, 5.0
    rho = delta + np.float32(1.0)
    np.testing.assert_allclose(np.sort(clump_mean_overdensity(mask, rho, mesh)), [2.0, 5.0])


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
        rho, jaxweb.maximum_signature(rho, RS, mesh=mesh), overdensity_threshold=20, mesh=mesh
    )
    assert summary["S_fil_thresh"] == want_summary["S_fil_thresh"]
    assert summary["S_wall_thresh"] == want_summary["S_wall_thresh"]
    # Signatures differ at float32 FFT precision, which flips the odd voxel
    # that sits right at a threshold.
    assert np.mean(np.asarray(got) != want) < 1e-4
