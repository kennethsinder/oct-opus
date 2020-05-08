#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-vengu
#SBATCH --time=03-00:00
#SBATCH --mem=127000M
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:2

cd $SLURM_TMPDIR
# No longer using original (non-flattened) data for now:
# tar --use-compress-program=pigz -xf ~/projects/def-vengu/s2saberi/all_data_original.tar.gz
unzip ~/projects/def-vengu/s2saberi/all_data_flattened.zip

cd -
module load cuda/10.0
./run.sh "$@" $SLURM_TMPDIR/all_data_flattened

