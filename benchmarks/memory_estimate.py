"""
Per-device memory of one jaxweb scale step, from XLA's buffer assignment.

Compiles the step for abstract inputs (nothing is allocated), so it runs on
a laptop for any grid size. Use simulated devices to model a GPU node:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        python benchmarks/memory_estimate.py 2048

The peaks include the density, which callers usually keep on the device for
tagging. At 2048³ on 4 devices the CPU compiler gives the same buffer
assignment as XLA GPU, and the peaks match the ones measured on 4× A100 80 GB
(65.3 GiB, or 73.3 GiB for NEXUS+ without offload). Run it on the GPUs
themselves to be sure after changing ``fft.py``, ``hessian.py`` or
``signatures.py``: XLA GPU adds transposes the CPU compiler does not.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

import jaxweb
from jaxweb.mesh import CHUNK_ELEMENTS, kspace_sharding, real_sharding, sig_sharding
from jaxweb.signatures import _make_base, _make_loggauss_step, _make_signature_step

GiB = 2**30


def _footprint(compiled):
    """Arguments + temporaries + outputs that do not alias an argument, in bytes."""
    mem = compiled.memory_analysis()
    return (
        mem.argument_size_in_bytes
        + mem.temp_size_in_bytes
        + mem.output_size_in_bytes
        - mem.alias_size_in_bytes
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="grid size per side")
    parser.add_argument("--algorithm", default="NEXUSPLUS", choices=["NEXUS", "NEXUSPLUS"])
    parser.add_argument("--chunk-elements", type=int, default=CHUNK_ELEMENTS)
    args = parser.parse_args()

    mesh = jaxweb.make_mesh()
    n, n_dev = args.n, len(jax.devices())
    chunk = args.chunk_elements
    shape = (n, n, n)
    slab = 4 * n**3 / n_dev
    real = jax.ShapeDtypeStruct(shape, jnp.float32, sharding=real_sharding(mesh))
    sigmax = jax.ShapeDtypeStruct((3, *shape), jnp.float32, sharding=sig_sharding(mesh))
    spec = jax.ShapeDtypeStruct((n, n, n // 2 + 1), jnp.complex64, sharding=kspace_sharding(mesh))
    scalar = jax.ShapeDtypeStruct((), jnp.float32)
    spectrum = 8 * n * n * (n // 2 + 1) / n_dev
    smooth = args.algorithm == "NEXUS"

    # The density stays on the device throughout (the caller holds it).
    base_stage = _footprint(
        _make_base(mesh, shape, args.algorithm, 1e-16, chunk).lower(real).compile()
    )
    step = _make_signature_step(mesh, shape, chunk, smooth=smooth)
    signature_stage = _footprint(step.lower(sigmax, spec, scalar).compile()) + slab

    print(f"{args.algorithm} {n}^3 on {n_dev} devices, per device (1 slab = {slab / GiB:.2f} GiB):")
    if smooth:
        peak = max(base_stage, signature_stage)
        print(
            f"  base {base_stage / slab:.1f}, signatures {signature_stage / slab:.1f} slabs"
            f" -> peak {peak / GiB:.2f} GiB"
        )
        return

    smooth_step = _make_loggauss_step(mesh, shape, chunk)
    # The density and the running max (3 slabs) stay on the device meanwhile.
    smoothing_stage = _footprint(smooth_step.lower(spec, scalar, scalar).compile()) + 4 * slab
    # Without offload the log-spectrum also stays during the signature stage.
    for offload, extra in ((False, spectrum), (True, 0)):
        peak = max(base_stage, smoothing_stage, signature_stage + extra)
        print(
            f"  offload={offload!s:5}: base {base_stage / slab:.1f}, smoothing "
            f"{smoothing_stage / slab:.1f}, signatures {(signature_stage + extra) / slab:.1f}"
            f" slabs -> peak {peak / GiB:.2f} GiB"
        )


if __name__ == "__main__":
    main()
