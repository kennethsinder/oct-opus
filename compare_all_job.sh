#!/bin/bash
#SBATCH --account=def-vengu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

cd $SLURM_TMPDIR
tar --use-compress-program=pigz -xf ~/projects/def-vengu/s2saberi/all_data_enfaces.tar.gz

cd -
module load cuda/10.0
python compare_all.py $SLURM_TMPDIR $@
