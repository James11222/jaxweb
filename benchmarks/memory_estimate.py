"""
Per-device memory of one jaxweb scale step, from XLA's buffer assignment.

Compiles the step for abstract inputs (nothing is allocated), so it runs on
a laptop for any grid size. Use simulated devices to model a GPU node:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        python benchmarks/memory_estimate.py 2048

The numbers come from the CPU compiler, so treat them as an estimate of the
GPU peak: cuFFT work areas and NCCL buffers are not included.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

import jaxweb
from jaxweb.mesh import CHUNK_ELEMENTS, kspace_sharding, sig_sharding
from jaxweb.signatures import _make_loggauss_step, _make_signature_step

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
    shape = (n, n, n)
    slab = 4 * n**3 / n_dev
    sigmax = jax.ShapeDtypeStruct((3, *shape), jnp.float32, sharding=sig_sharding(mesh))
    spec = jax.ShapeDtypeStruct((n, n, n // 2 + 1), jnp.complex64, sharding=kspace_sharding(mesh))
    scalar = jax.ShapeDtypeStruct((), jnp.float32)
    smooth = args.algorithm == "NEXUS"

    step = _make_signature_step(mesh, shape, args.chunk_elements, smooth=smooth)
    signature_stage = _footprint(step.lower(sigmax, spec, scalar).compile())

    print(f"{args.algorithm} {n}^3 on {n_dev} devices, per device (1 slab = {slab / GiB:.2f} GiB):")
    if smooth:
        print(f"  peak: {signature_stage / slab:.1f} slabs = {signature_stage / GiB:.2f} GiB")
        return

    smooth_step = _make_loggauss_step(mesh, shape)
    # The running max (3 slabs) stays on the device while the smoothing runs.
    smoothing_stage = _footprint(smooth_step.lower(spec, scalar, scalar).compile()) + 3 * slab
    # Without offload the log-spectrum (1 slab) also stays during the signature stage.
    spectrum = 8 * n * n * (n // 2 + 1) / n_dev
    for offload, extra in ((False, spectrum), (True, 0)):
        peak = max(smoothing_stage, signature_stage + extra)
        print(
            f"  offload={offload!s:5}: smoothing {smoothing_stage / slab:.1f}, signatures "
            f"{(signature_stage + extra) / slab:.1f} slabs -> peak {peak / GiB:.2f} GiB"
        )


if __name__ == "__main__":
    main()
