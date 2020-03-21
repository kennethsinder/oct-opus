#!/bin/bash

if [ "$1" = "" ];then
    NUM_EPOCHS=10
else
    NUM_EPOCHS=$1
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

python run_cnn.py train $HARDWARE -d $DATA_DIR -e $NUM_EPOCHS
