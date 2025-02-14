#!/bin/bash

if [[ "${1}" == "cluster" ]]
    then
        for entry in $(squeue --me --noheader --format="%i|%j|%N")
        do
            id=$(echo $entry | awk -F'|' '{print $1}')
            name=$(echo $entry | awk -F'|' '{print $2}')
            node=$(echo $entry | awk -F'|' '{print $4}')

            echo "Abort id=${id}, name=${name}, node=${node}."
            scancel --me $id ; srun --pty -w $node ${BASH_SOURCE[0]} node
        done
fi

if [[ "${1}" == "node" ]]
    then
        export SCRATCH_MNT="/mnt/scratch/"
        export WIP_PATH="/tmp/rllib_gazebo"
        export RES_FILE="${HOSTNAME}.tar.gz"

        if [[ -d $WIP_PATH ]]
            then
                cp -r /tmp/ray $WIP_PATH/ray
                cp -r ~/.ros $WIP_PATH/ros
                cp -r ~/.gz $WIP_PATH/gz

                tar -czf /tmp/$RES_FILE $WIP_PATH/*
                rsync -a /tmp/$RES_FILE ~/$RES_FILE || rsync -a /tmp/$RES_FILE $SCRATCH_MNT/$RES_FILE
        fi
fi
