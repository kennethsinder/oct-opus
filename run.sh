#!/bin/bash

if [ "$1" = "" ];then
    STARTING_EPOCH=1
else
    STARTING_EPOCH=$1
fi

if [ "$2" = "" ];then
    HARDWARE="gpu"
else
    HARDWARE=$2
fi

if [ "$3" = "" ];then
    DATA_DIR="~/projects/def-vengu/s2saberi"
else
    DATA_DIR=$3
fi

LOGDIR='logs/'$(date +"%d-%m-%Y_%H:%M:%S")

echo "Logs are being sent to $LOGDIR..."
python run.py --restore --logdir $LOGDIR --epoch $STARTING_EPOCH --datadir $DATA_DIR train $HARDWARE

