#!/bin/bash
#SBATCH --account=def-vengu
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

cd $SLURM_TMPDIR
tar -xf --use-compress-program=pigz ~/projects/def-vengu/s2saberi/all_data_original.tar.gz
tar -xf --use-compress-program=pigz ~/projects/def-vengu/s2saberi/all_data_enface.tar.gz

cd -
module load cuda/10.0
python compare_all.py $SLURM_TMPDIR/all_data_original
