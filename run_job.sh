#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-vengu
#SBATCH --time=03-00:00
#SBATCH --mem=127000M
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:2

# Example Usage on Sharcnet: `sbatch run_job.sh 2 -k` for 2 epochs per fold training * 5 folds
# (K-folds cross validation)
# Any command-line arguments after the start and end epoch numbers are passed along faithfully
# to the `cgan.py` script.

cd $SLURM_TMPDIR
tar --use-compress-program=pigz -xf ~/projects/def-vengu/s2saberi/single_poly_flattened_normalized.tar.gz

cd -
module load cuda/10.0
./run.sh --num-epochs "$1" --datadir "$SLURM_TMPDIR"/single_poly_flattened_normalized "${@:2}" train
