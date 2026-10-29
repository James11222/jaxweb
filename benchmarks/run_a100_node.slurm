#!/bin/bash
# One node, 4x A100 80 GB, one process driving all 4 GPUs.
# Usage (from the repository root): sbatch benchmarks/run_a100_node.slurm [path/to/density.npy]
# The default input is the L1_m9_DMO snapshot 77 cube made by
#     python benchmarks/mass_to_density.py ../data/dm_mass_field_L1_m9_DMO_0077.hdf5 \
#         ../data/dm_density_L1_m9_DMO_0077.npy
# Needs a conda environment with `pip install -e ".[gpu]"`; set its path at `conda activate` below.
#SBATCH --job-name=jaxweb
#SBATCH --account=jdunkley
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
# Della's gpu partition also has 40 GB A100s and MIG slices; gpu80 selects the 4x A100 80 GB nodes.
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00

set -eo pipefail  # no -u: conda activate reads unset variables
density=${1:-../data/dm_density_L1_m9_DMO_0077.npy}
if [[ ! -f $density ]]; then
    echo "no density cube at $density" >&2
    echo "usage: sbatch benchmarks/run_a100_node.slurm [path/to/density.npy]" >&2
    exit 1
fi

module purge
module load anaconda3/2026.7
conda activate /scratch/gpfs/TEYSSIER/js7501/envs/jaxweb

# Let XLA use (almost) all of each GPU, allocate on demand from CUDA's
# stream-ordered pool, and avoid the fragmentation of the default allocator.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async

nvidia-smi --query-gpu=name,memory.total --format=csv
srun python benchmarks/run_signatures.py "$density"
