#!/bin/bash

pids=()
for i in {0..5}
do
   python train.py --config=configs/xql.yaml --seed=$i --path BG_Online_Normal_1.0TargetBC &

    pids+=( "$!" )
    sleep 5 # add 5 second delay
done

for pid in "${pids[@]}"; do
    wait $pid
done

# sleep 2d
