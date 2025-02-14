#!/bin/bash

PATHS=(
    "HOME_PATHS"
    "TEMP_PATHS"
    "PROJ_PATHS"
    "EXP_PATHS"
    "WS_PATHS"
    "GIT_PATHS"
)

HOME_PATHS=(
    "~/.colcon"
    "~/.ros"
    "~/.ign"
    "~/.gz"
    "~/.sdformat"
    "~/.keras"
    "~/.singularity"
    "~/.local"
    "~/.nv"
)

TEMP_PATHS=(
    "/tmp/*wandb*"
    "/tmp/*ray*"
    "/tmp/*rllib*"
)

PROJ_PATHS=(
    "~/*wandb*"
    "~/*ray*"
    "~/*rllib*"
    "~/latest"
    "~/crl"
)

EXP_PATHS=(
    "~/git/egge/generated"
)

WS_PATHS=(
    "~/git/icrl/workspace/base/build"
    "~/git/icrl/workspace/base/install"
    "~/git/icrl/workspace/base/log"
    "~/git/icrl/workspace/devel/build"
    "~/git/icrl/workspace/devel/install"
    "~/git/icrl/workspace/devel/log"
)

GIT_PATHS=(
    "~/git/icrl"
    "~/git/sccl"
    "~/git/egge"
)

for path in ${PATHS[@]}
do
    read -p "Starting with ${path}? (y/n): " choice

    if [[ "${choice}" != "y" && "${choice}" != "Y" ]]
        then continue
    fi

    ENTRIES=${!path}
    for entry in ${ENTRIES[@]}
    do
        if [[ -e $entry ]]
            then
                base_path=$(realpath -s $path)
                echo "Cleanup ${base_path}..."
                rm -Ir $base_path
        fi
    done
done
