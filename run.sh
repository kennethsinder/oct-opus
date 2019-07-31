#!/bin/bash

STARTING_EPOCH=1
ENDING_EPOCH=2
LOGDIR='logs/'$(date +"%d-%m-%Y_%H:%M:%S")
CURR_STEP=1

echo "Logs are being sent to $LOGDIR..."
for i in $( seq $STARTING_EPOCH $ENDING_EPOCH )
do
    echo "----- Epoch number $i -----"
    python run.py -r -s $CURR_STEP train $LOGDIR
    CURR_STEP=$?
done
