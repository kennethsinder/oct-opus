#!/bin/bash

if [ "$1" = "" ];then
    STARTING_EPOCH=1
else
    STARTING_EPOCH=$1
fi

if [ "$2" = "" ];then
    ENDING_EPOCH=100
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
for i in $( seq $STARTING_EPOCH $ENDING_EPOCH )
do
    echo "----- Epoch number $i -----"
    python run.py --restore --logdir $LOGDIR --epoch $i -d $DATA_DIR train $HARDWARE
    n=$(($i%5))
    if [[ n -eq 0 ]]; then
        python run.py --restore --epoch $i --datadir $DATA_DIR predict $HARDWARE
    fi
done

