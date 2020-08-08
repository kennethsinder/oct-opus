#!/bin/bash
#SBATCH --account=def-vengu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

# NOTE: Change `def-vengu` to the appropriate Sharcnet account.
#
# Usage: `sbatch ./scripts/compare_all_job.sh <experiment_dir>`
# The `experiment_dir` is passed faithfully along to `compare_all.py`.
# The flattened sedated rat data `all_data_flattened.zip` 
# is assumed to be one level up from the execution of this script.
# It is extracted for you and that extract
# location is fed into the `compare_all.py` script so you don't have to do
# that work on Sharcnet/Graham. `compare_all.py` is assumed to be in
# the `./scripts` folder.

cwd=$(pwd)
cd "$SLURM_TMPDIR"
tar --use-compress-program=pigz -xf "$cwd"/../single_poly_flattened_normalized.tar.gz

cd -
module load cuda/10.0
python scripts/compare_all.py "$1" "$SLURM_TMPDIR"/single_poly_flattened_normalized
