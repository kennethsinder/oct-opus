#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-vengu
#SBATCH --time=01-00:00
#SBATCH --mem=127000M
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:2

cd $SLURM_TMPDIR

tar --use-compress-program=pigz -xf ~/projects/def-vengu/s2saberi/all_data_reflattened.tar.gz
DATA_DIR=$SLURM_TMPDIR/all_data_reflattened

cd -
module load cuda/10.0

python gen_initial_cnn_weights.py gpu -d $DATA_DIR "${@:2}"
