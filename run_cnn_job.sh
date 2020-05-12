#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-vengu
#SBATCH --time=03-00:00
#SBATCH --mem=127000M
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:2

cd $SLURM_TMPDIR

tar --use-compress-program=pigz -xf ~/projects/def-vengu/s2saberi/all_data_original.tar.gz
DATA_DIR=$SLURM_TMPDIR/all_data_original

cd -
module load cuda/10.0

if [ "$1" = "" ];then
	MODE="train"
else
	MODE=$1
fi

if [ "$2" = "" ];then
	EPOCHS_FLAG=""
else
	EPOCHS_FLAG="-e $2"
fi

if [ "$3" = "" ];then
	CHECKPOINTS_FLAG=""
else
	CHECKPOINTS_FLAG="-c $3"
fi

python cnn.py $MODE gpu -d $DATA_DIR $EPOCHS_FLAG $CHECKPOINTS_FLAG
