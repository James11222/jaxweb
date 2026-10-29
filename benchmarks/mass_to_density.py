"""
Turn a gridded mass field in an HDF5 file into a δ+1 = ρ/⟨ρ⟩ ``.npy`` cube.

    python benchmarks/mass_to_density.py mass.hdf5 density.npy [--dataset MassField]
        [--downsample 2]

Streams over x-planes, so memory stays at one block (``--planes`` x-planes)
whatever the grid size. ``--downsample k`` sums the mass in k³ cells first,
so the coarse cube conserves mass. The mean is accumulated in float64.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np


def _coarsen(chunk, k):
    """Sum ``chunk`` over k³ cells."""
    if k == 1:
        return chunk
    a, b, c = (s // k for s in chunk.shape)
    return chunk.reshape(a, k, b, k, c, k).sum(axis=(1, 3, 5), dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--dataset", default="MassField")
    parser.add_argument("--planes", type=int, default=64, help="x-planes per block")
    parser.add_argument("--downsample", type=int, default=1, help="coarsening factor k")
    args = parser.parse_args()

    k = args.downsample
    with h5py.File(args.source, "r") as handle:
        mass = handle[args.dataset]
        if any(s % k for s in mass.shape) or args.planes % k:
            msg = f"--downsample {k} must divide the shape {mass.shape} and --planes"
            raise ValueError(msg)
        n = mass.shape[0]
        blocks = [slice(i, min(i + args.planes, n)) for i in range(0, n, args.planes)]
        shape = tuple(s // k for s in mass.shape)

        total = 0.0
        for block in blocks:
            total += mass[block].sum(dtype=np.float64)
        mean = total / np.prod(shape, dtype=np.float64)
        if not np.isfinite(mean) or mean <= 0:
            msg = f"bad mass field: mean {mean}"
            raise ValueError(msg)

        out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float32, shape=shape)
        lo, hi = np.inf, -np.inf
        for block in blocks:
            density = (_coarsen(mass[block], k) / mean).astype(np.float32)
            lo, hi = min(lo, density.min()), max(hi, density.max())
            out[block.start // k : block.stop // k] = density
        out.flush()

    if lo < 0:
        msg = f"negative mass in {args.source}: min δ+1 = {lo}"
        raise ValueError(msg)
    print(f"{args.source}:{args.dataset} -> {args.output} {shape}")
    print(f"  mean mass per cell {mean:.6e}; delta+1 in [{lo:.3e}, {hi:.3e}]")


if __name__ == "__main__":
    main()
