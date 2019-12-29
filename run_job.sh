#!/bin/bash
#SBATCH --account=def-vengu
#SBATCH --time=03-00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

cd $SLURM_TMPDIR
tar -xzf ~/projects/def-vengu/s2saberi/*.tar.gz

cd -
./run.sh "$@" $SLURM_TMPDIR

