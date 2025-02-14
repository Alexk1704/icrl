#!/bin/bash

#############################################################################################################################
# This script is the central component, which will be invoked by the user or any workload manager.                          #
# Here, all further components are invoked, depending on the provieded arguments.                                           #
# Moreover, some information about the shell, the caller, the calle etc. are collected.                                     #
# Finally, a trap is set to catch all signals and each util script is loaded.                                               #
#############################################################################################################################

CNT="$#"
FLAGS="$-"

ARGS=("$@")
CWD="${PWD}"

#----------------------------------------------------------------------------------------------------------------------------

CALLER_SOURCE=$(realpath "${0}")
CALLER_PATH=$(dirname "${CALLER_SOURCE}")
CALLER_NAME=$(basename "${CALLER_SOURCE}")

CALLEE_SOURCE=$(realpath "${BASH_SOURCE[0]}")
CALLEE_PATH=$(dirname "${CALLEE_SOURCE}")
CALLEE_NAME=$(basename "${CALLEE_SOURCE}")

SHELL_PROCESS=$(ps -p $$ | tail -1 | awk '{print $NF}')

#----------------------------------------------------------------------------------------------------------------------------

if [[ ! -v EXIT_CODE ]] ; then export EXIT_CODE="" ; fi
trap "echo -e '!!!THIS EXIT IS NOT INTENDED!!!\nDO NOT ABORT THE EXECUTION NASTY; THIS CAN LEAD TO WEIRD SIDEFFECTS...'" 0
# SIGNAL 0 is a bad choice, since it will end always the execution....

#----------------------------------------------------------------------------------------------------------------------------

echo "Script ${CALLEE_SOURCE} invoked via $(cat /proc/${PPID}/comm) with ${CNT} arguments by ${CALLER_SOURCE}!"

export COMPONENTS=("$(ls ${CALLEE_PATH}/components/*.bash)")

# somehow define default invocations within the corresponding config?
MIN=("ENVIRON" "PREPARE" "BUILD" "DEPLOY" "CHECK" "EXECUTE" "BACKUP" "BUNDLE")
FULL=("ENVIRON" "PREPARE" "BUILD" "CLONE" "DEPLOY" "CHECK" "EXECUTE" "BACKUP" "WIPE" "AWAIT" "BUNDLE" "RELEASE" "REPLACE" "CLEANUP")

if [[ -z $ARGS ]]
    # then ARGS=("${MIN[@]}")
    then ARGS=("${FULL[@]}")
fi

#----------------------------------------------------------------------------------------------------------------------------

# source "${CALLEE_PATH}/configs/default.bash" # use this for debugging
# source "${CALLEE_PATH}/configs/bash.bash" # use this for your machine
source "${CALLEE_PATH}/configs/slurm.bash" # use this for the cluster

source "${CALLEE_PATH}/utils/template.bash"
source "${CALLEE_PATH}/utils/functions.bash"
source "${CALLEE_PATH}/utils/variables.bash"

apply_config "COMMON"

#----------------------------------------------------------------------------------------------------------------------------

for arg in ${ARGS[@]}
do
    MATCHES=("$(filter_matches COMPONENTS array ${arg,,})")
    valid=$(check_length "MATCHES" "array" "1")

    if $valid
        then
            if [[ -v DEBUG ]] ; then read -p "Run component ${MATCHES[*]}?" skip ; fi

            print_header "${MATCHES[*]}"
            enter_component "${MATCHES[*]}"
            time source "${MATCHES[*]}"
            exit_component "${MATCHES[*]}"
            print_footer "${MATCHES[*]}"
        else
            echo "Found no matching script for argument ${arg}..."
    fi
done

#----------------------------------------------------------------------------------------------------------------------------

trap - 0
if [[ $0 == ${BASH_SOURCE[0]} ]] ; then exit $EXIT_CODE ; fi
