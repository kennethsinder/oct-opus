#!/bin/bash

if [ "$1" = "" ];then
    STARTING_EPOCH=1
else
    STARTING_EPOCH=$1
fi

if [ "$2" = "" ];then
    ENDING_EPOCH=10
else
    ENDING_EPOCH=$2
fi

if [ "$3" = "" ];then
    HARDWARE="gpu"
else
    HARDWARE=$3
fi

if [ "$4" = "" ];then
    DATA_DIR="~/projects/def-vengu/s2saberi"
else
    DATA_DIR=$4
fi

LOGDIR='logs/'$(date +"%d-%m-%Y_%H:%M:%S")

echo "Logs are being sent to $LOGDIR..."
python run.py --logdir $LOGDIR --starting-epoch $STARTING_EPOCH --ending-epoch $ENDING_EPOCH --datadir $DATA_DIR train $HARDWARE

