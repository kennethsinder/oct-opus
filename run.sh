#!/bin/bash

NUM_EPOCHS=10

python run.py train 0

STARTING_EPOCH=1
for i in $( seq $STARTING_EPOCH $NUM_EPOCHS )
do
    python run.py restore train $i
done
