#!/bin/bash
#SBATCH --account=def-vengu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

# NOTE: Change `def-vengu` to the appropriate Sharcnet account.
# Usage: `sbatch compare_all_job.sh <experiment_dir>`
# The `experiment_dir` is passed faithfully along to `compare_all.py`.
# The flattened sedated rat data is assumed to be *two* levels up from
# the location of this script, and is extracted for you and that extract
# location is fed into the `compare_all.py` script so you don't have to do
# that work on Sharcnet/Graham.

cwd=$(pwd)
cd "$SLURM_TMPDIR"
tar --use-compress-program=pigz -xf "$cwd"/../../all_data_flattened.tar.gz

cd -
module load cuda/10.0
python compare_all.py "$1" "$SLURM_TMPDIR"/all_data_flattened
