#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-vengu
#SBATCH --time=03-00:00
#SBATCH --mem=127000M
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:2

# Example Usage on Sharcnet: `sbatch run_job.sh 1 2` for 2 epochs per fold training * 5 folds
# Any command-line arguments after the start and end epoch numbers are passed along faithfully
# to the `cgan.py` script.

cd $SLURM_TMPDIR
# No longer using original (non-flattened) data for now:
# tar --use-compress-program=pigz -xf ~/projects/def-vengu/s2saberi/all_data_original.tar.gz
unzip ~/projects/def-vengu/s2saberi/all_data_flattened.zip

cd -
module load cuda/10.0
./run.sh --starting-epoch "$1" --ending-epoch "$2" --datadir "$SLURM_TMPDIR"/all_data_flattened "${@:3}" train
