#!/bin/bash

source=~/Desktop/Experiments/v3/CURRENT_DUMP/VALIDATE__24-03-19-14-18-58/results/failed
target=~/Downloads/bench/logs

for archive in $(ls ${source})
do
    echo "extracting from ${archive}..."

    directory=${archive%%.*}
    mkdir -p ${target}/${directory}
    tar -xzf ${source}/${archive} -C ${target}/${directory} ./logs/
    mv ${target}/${directory}/logs/* ${target}/${directory}/
    rmdir ${target}/${directory}/logs
done
