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

LOGDIR='logs/'$(date +"%d-%m-%Y_%H:%M:%S")

echo "Logs are being sent to $LOGDIR..."
for i in $( seq $STARTING_EPOCH $ENDING_EPOCH )
do
    echo "----- Epoch number $i -----"
    python run.py --restore --logdir $LOGDIR --epoch $i train $HARDWARE
    n=$(($i%5))
    if [[ n -eq 0 ]]; then
        python run.py --restore --epoch $i predict $HARDWARE && python plot.py ./predicted/2015-10-22___512_2048_Horizontal_Images13 $i
    fi
done
