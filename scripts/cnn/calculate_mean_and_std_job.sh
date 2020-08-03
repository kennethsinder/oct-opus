#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-vengu
#SBATCH --time=03-00:00
#SBATCH --mem=127000M
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:2

cd $SLURM_TMPDIR

tar --use-compress-program=pigz -xf ~/projects/def-vengu/s2saberi/single_poly_flattened.tar.gz
DATA_DIR=$SLURM_TMPDIR/single_poly_flattened

K=5
SELECTED=${1}
SEED=42

cd -
module load cuda/10.0

python calculate_mean_and_std.py -d $DATA_DIR -k $K -s $SELECTED -sd $SEED
