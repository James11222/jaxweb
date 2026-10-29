"""Time ``maximum_signature`` on a density cube and report peak device memory.

python benchmarks/run_signatures.py density.npy [--algorithm NEXUS]
"""

from __future__ import annotations

import argparse
import time

import jax

import jaxweb

GiB = 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--algorithm", default="NEXUSPLUS", choices=["NEXUS", "NEXUSPLUS"])
    parser.add_argument("--n-scales", type=int, default=9)
    parser.add_argument("--offload", choices=["auto", "on", "off"], default="auto")
    args = parser.parse_args()

    jaxweb.initialize()
    mesh = jaxweb.make_mesh()
    Rs = [2 ** (n / 2) for n in range(args.n_scales)]
    offload = {"auto": None, "on": True, "off": False}[args.offload]

    t0 = time.perf_counter()
    rho = jaxweb.load_density(args.path, mesh)
    rho.block_until_ready()
    t1 = time.perf_counter()
    sigmax = jaxweb.maximum_signature(rho, Rs, algorithm=args.algorithm, mesh=mesh, offload=offload)
    sigmax.block_until_ready()
    t2 = time.perf_counter()

    if jax.process_index() == 0:
        print(f"{args.algorithm} {rho.shape} on {len(jax.devices())} devices, {len(Rs)} scales")
        print(f"  load {t1 - t0:.1f} s, signatures {t2 - t1:.1f} s")
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        if "peak_bytes_in_use" in stats:
            print(f"  {device}: peak {stats['peak_bytes_in_use'] / GiB:.1f} GiB")


if __name__ == "__main__":
    main()
