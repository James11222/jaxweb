#!/bin/bash
# Signatures + tagging of a density cube on one node, 4x A100 80 GB, one process.
# Usage (from the repository root): sbatch scripts/tag_cube.slurm [path/to/density.npy] [extra args]
# Writes <stem>_NEXUSPLUS_structure_filter.npy and <stem>_NEXUSPLUS_summary.json
# next to the cube; see scripts/tag_cube.py for options (--out-dir, --algorithm, ...).
#SBATCH --job-name=jaxweb-tag
#SBATCH --account=jdunkley
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
# Della's gpu partition also has 40 GB A100s and MIG slices; gpu80 selects the 4x A100 80 GB nodes.
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
# Tagging has not been timed at 2048^3 yet; signatures take ~1.5 min including loading.
#SBATCH --time=00:20:00

set -eo pipefail  # no -u: conda activate reads unset variables
density=${1:-../data/dm_density_L1_m9_DMO_0077.npy}
shift || true
if [[ ! -f $density ]]; then
    echo "no density cube at $density" >&2
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
srun python scripts/tag_cube.py "$density" "$@"
