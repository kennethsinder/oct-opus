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

DATA_DIR=$3

python cgan.py --starting-epoch $STARTING_EPOCH --ending-epoch $ENDING_EPOCH --datadir $DATA_DIR "$@" train

