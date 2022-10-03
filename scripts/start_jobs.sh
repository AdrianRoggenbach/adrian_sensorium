#!/bin/bash

filename=$1

# read each line of filename and execute sbatch command
while read line || [ -n "$line" ]; do

echo Adding to queue: $line
sbatch -J "M: $line" schedule_one_model.batch $line
sleep 2
done < $filename

# check queue with: squeue -u ${USER}