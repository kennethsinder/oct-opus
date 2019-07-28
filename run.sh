#!/bin/bash

STARTING_EPOCH=1
ENDING_EPOCH=100
for i in $( seq $STARTING_EPOCH $ENDING_EPOCH )
do
    echo "----- Epoch number $i -----"
    python run.py restore train
done
