#!/bin/bash
#SBATCH --account=def-vengu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

python post-normalizer.py "$1"
