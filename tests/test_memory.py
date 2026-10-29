"""Guard the per-device memory budget that lets 2048³ fit on 4× 80 GB GPUs.

XLA's buffer assignment for the compiled steps, at 256³ on 4 devices with the
chunking scaled so the loops run as many chunks as at 2048³. The numbers are
in field slabs (one float32 x-slab per device). Run under ``JAX_PLATFORMS=cuda``
on a 4-GPU node they check the GPU compiler, whose transposes are what broke
the budget before (see ``jaxweb.fft._row_major``).

At 2048³ on 4× A100 80 GB the measured peak is 65.3 GiB (~8.2 slabs) for
both NEXUS and NEXUS+ (with the log-spectrum offloaded), with the caller
keeping the density on the device.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxweb.mesh import kspace_sharding, real_sharding, sig_sharding
from jaxweb.signatures import _make_base, _make_loggauss_step, _make_signature_step

N = 256
CHUNK = N * N  # one x-plane / one k_y column per chunk, 64 chunks per device


def footprint(compiled):
    mem = compiled.memory_analysis()
    return (
        mem.argument_size_in_bytes
        + mem.temp_size_in_bytes
        + mem.output_size_in_bytes
        - mem.alias_size_in_bytes
    )


@pytest.fixture(scope="module")
def specs(mesh):
    shape = (N, N, N)
    sig = jax.ShapeDtypeStruct((3, *shape), jnp.float32, sharding=sig_sharding(mesh))
    spec = jax.ShapeDtypeStruct((N, N, N // 2 + 1), jnp.complex64, sharding=kspace_sharding(mesh))
    scalar = jax.ShapeDtypeStruct((), jnp.float32)
    slab = 4 * N**3 / 4
    return shape, sig, spec, scalar, slab


@pytest.mark.parametrize(("smooth", "budget"), [(True, 7.5), (False, 7.5)])
def test_signature_stage_budget(mesh, specs, smooth, budget):
    shape, sig, spec, scalar, slab = specs
    step = _make_signature_step(mesh, shape, CHUNK, smooth=smooth)
    assert footprint(step.lower(sig, spec, scalar).compile()) / slab < budget


def test_loggauss_stage_budget(mesh, specs):
    shape, _, spec, scalar, slab = specs
    step = _make_loggauss_step(mesh, shape, CHUNK)
    # Log-spectrum in, spectrum out, one spectrum-sized buffer in between;
    # plus 3 slabs of running max that stay on the device meanwhile.
    assert footprint(step.lower(spec, scalar, scalar).compile()) / slab + 3 < 6.5


@pytest.mark.parametrize("algorithm", ["NEXUS", "NEXUSPLUS"])
def test_base_stage_budget(mesh, specs, algorithm):
    shape, _, _, _, slab = specs
    real = jax.ShapeDtypeStruct(shape, jnp.float32, sharding=real_sharding(mesh))
    base = _make_base(mesh, shape, algorithm, 1e-16, CHUNK)
    # Density in, spectrum out, one spectrum-sized buffer in between.
    assert footprint(base.lower(real).compile()) / slab < 3.5
