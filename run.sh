#!/bin/bash

NUM_EPOCHS=20

python run.py train 7

STARTING_EPOCH=8
for i in $( seq $STARTING_EPOCH $NUM_EPOCHS )
do
    python run.py restore train $i
done
