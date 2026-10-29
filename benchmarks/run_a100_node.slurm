#!/bin/bash
# One node, 4x A100 80 GB, one process driving all 4 GPUs.
# Usage: sbatch benchmarks/run_a100_node.slurm path/to/density.npy
#SBATCH --job-name=jaxweb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00

# Let XLA use (almost) all of each GPU, allocate on demand from CUDA's
# stream-ordered pool, and avoid the fragmentation of the default allocator.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async

srun python benchmarks/run_signatures.py "$1"
