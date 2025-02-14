#!/bin/bash

declare -A REPOS=(
    ["icrl"]="main"
    ["sccl"]="main"
    ["egge"]="main"
)

declare -A SSH_REPOS=(
    ["icrl"]="git@github.com:BenediktBagus/icrl.git"
    ["sccl"]="git@github.com:Alexk1704/sccl.git"
    ["egge"]="git@github.com:BenediktBagus/egge.git"
)

declare -A HTTPS_REPOS=(
    ["icrl"]="https://github.com/BenediktBagus/icrl.git"
    ["sccl"]="https://github.com/Alexk1704/sccl.git"
    ["egge"]="https://github.com/BenediktBagus/egge.git"
)

for repo in ${!REPOS[@]}
do
    cd ~/git

    if [[ ! -d ./$repo ]]
        then
            echo "Try cloning ${repo}..."
            { git clone ${SSH_REPOS[$repo]} || git clone ${HTTPS_REPOS[$repo]} ;} && echo "success" || echo "failed"
    fi

    cd ~/git/$repo

    if [[ "${1}" =~ "force" ]]
        then
            echo "Try resetting ${REPOS[$repo]} of ${repo}..."
            git reset --hard && echo "success" || echo "failed"
        else
            echo "Try stashing ${REPOS[$repo]} of ${repo}..."
            git stash --all && echo "success" || echo "failed"
    fi

    echo "Try pulling ${REPOS[$repo]} of ${repo}..."
    git pull origin/${REPOS[$repo]} && echo "success" || echo "failed"
done
