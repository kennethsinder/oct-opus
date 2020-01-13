#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-vengu
#SBATCH --time=03-00:00
#SBATCH --mem=127000M
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:2

cd $SLURM_TMPDIR
tar -xzf ~/projects/def-vengu/s2saberi/all_data_original.tar.gz
tar -xzf ~/projects/def-vengu/s2saberi/all_data_enface.tar.gz

cd -
module load cuda/10.0
./run.sh "$@" $SLURM_TMPDIR
