from __future__ import annotations

import numpy as np
import pytest

import jaxweb
from jaxweb.fft import largest_divisor_at_most


@pytest.mark.parametrize("shape", [(32, 32, 32), (16, 32, 24)])
def test_rfft3_matches_numpy(mesh, shape):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    got = np.asarray(jaxweb.rfft3(x, mesh))
    want = np.fft.rfftn(x.astype(np.float64))
    assert got.dtype == np.complex64
    np.testing.assert_allclose(got, want, atol=2e-4 * np.abs(want).max())


def test_irfft3_round_trip(mesh):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 32, 32)).astype(np.float32)
    back = np.asarray(jaxweb.irfft3(jaxweb.rfft3(x, mesh), mesh, 32))
    assert back.dtype == np.float32
    np.testing.assert_allclose(back, x, atol=1e-5)


def test_bad_shapes_raise(mesh):
    with pytest.raises(ValueError, match="divisible"):
        jaxweb.maximum_signature(np.ones((30, 32, 32), np.float32), [1.0], mesh=mesh)
    with pytest.raises(ValueError, match="even"):
        jaxweb.maximum_signature(np.ones((32, 32, 31), np.float32), [1.0], mesh=mesh)


def test_largest_divisor_at_most():
    assert largest_divisor_at_most(512, 8) == 8
    assert largest_divisor_at_most(12, 5) == 4
    assert largest_divisor_at_most(7, 0) == 1


def test_chunked_transforms_match_one_chunk(mesh):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((32, 32, 32)).astype(np.float32)
    # 32·32 elements per x-plane and 32·17 per k_y column: 2**11 gives 4 chunks
    # of x-planes and 8 of k_y columns per device.
    spec = np.asarray(jaxweb.rfft3(x, mesh, chunk_elements=2**11))
    want = np.fft.rfftn(x.astype(np.float64))
    np.testing.assert_allclose(spec, want, atol=2e-4 * np.abs(want).max())
    back = np.asarray(jaxweb.irfft3(jaxweb.rfft3(x, mesh), mesh, 32, chunk_elements=2**11))
    np.testing.assert_allclose(back, x, atol=1e-5)
