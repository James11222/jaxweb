from __future__ import annotations

import os

# Four simulated CPU devices, so every test exercises real 4-way sharding and
# all-to-alls without a GPU. Must be set before JAX is imported.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import pytest

import jaxweb


@pytest.fixture(scope="session")
def mesh():
    assert len(jax.devices()) == 4
    return jaxweb.make_mesh()
