#!/bin/bash

#############################################################################################################################
# This component contains the wipe logic to remove unneccessary data over the execution-related paths.                      #
# Depending on the setup, the data and the paths can be selected.                                                           #
# This will be triggered per run and is equivalent to cleanup, which will be triggered per experiment.                      #
#############################################################################################################################

lookup="$_"
apply_config "${lookup^^}"


if [[ "${WIPE_CONFIG['TEMP']}" != "none" ]]
    then print_warn "wipe of TEMP will be performed!"
    else print_warn "wipe of TEMP will be skipped!"
fi

if [[ "${WIPE_CONFIG['ORG']}" != "none" ]]
    then print_warn "wipe of ORG_PATH will be performed!"
    else print_warn "wipe of ORG_PATH will be skipped!"
fi

if [[ "${WIPE_CONFIG['OPT']}" != "none" ]]
    then print_warn "wipe of OPT_PATH will be performed!"
    else print_warn "wipe of OPT_PATH will be skipped!"
fi

if [[ "${WIPE_CONFIG['TMP']}" != "none" ]]
    then print_warn "wipe of TMP_PATH will be performed!"
    else print_warn "wipe of TMP_PATH will be skipped!"
fi

if [[ "${WIPE_CONFIG['WIP']}" != "none" ]]
    then print_warn "wipe of WIP_PATH will be performed!"
    else print_warn "wipe of WIP_PATH will be skipped!"
fi

if [[ "${WIPE_CONFIG['RES']}" != "none" ]]
    then print_warn "wipe of RES_PATH will be performed!"
    else print_warn "wipe of RES_PATH will be skipped!"
fi

if [[ "${WIPE_CONFIG['PUB']}" != "none" ]]
    then print_warn "wipe of PUB_PATH will be performed!"
    else print_warn "wipe of PUB_PATH will be skipped!"
fi


function wipe_condition {
    match_any "${WIPE_CONFIG[${1}]}" "${@:2}"
    # do not use "" for the last parameter
    # otherwise it will be understood as a single string
}

function wipe_execute {
    print_info "wipe ${1} artifact files"
    eval_state $(protected_delete "${2}")
}

#----------------------------------------------------------------------------------------------------------------------------

wipe_condition "TEMP" "wandb" "all" &&
wipe_execute "wandb" "/tmp/*wandb*"

wipe_condition "TEMP" "ray" "all" &&
wipe_execute "ray" "${TUNE_TEMP_DIR}"

wipe_condition "TEMP" "ray" "all" &&
wipe_execute "ray result" "${TUNE_RESULT_DIR}"

wipe_condition "TEMP" "log" "all" &&
wipe_execute "ros log" "${HOME}/.ros"

wipe_condition "TEMP" "log" "all" &&
wipe_execute "gz log"  "${HOME}/.gz"

wipe_condition "ORG" "git" "all" &&
wipe_execute "git" "${GIT_PATH}/*"

wipe_condition "ORG" "sif" "all" &&
wipe_execute "sif" "${SIF_PATH}/*"

wipe_condition "OPT" "scm" "all" &&
wipe_execute "scm" "${SCM_PATH}/*"

wipe_condition "OPT" "var" "all" &&
wipe_execute "var" "${VAR_PATH}/*"

wipe_condition "TMP" "bld" "all" &&
wipe_execute "bld" "${BLD_PATH}/*"

wipe_condition "TMP" "jnk" "all" &&
wipe_execute "jnk" "${JNK_PATH}/*"

wipe_condition "WIP" "src" "all" &&
wipe_execute "src" "${SRC_PATH}/*"

wipe_condition "WIP" "dat" "all" &&
wipe_execute "dat" "${DAT_PATH}/*"

wipe_condition "RES" "dmp" "all" &&
wipe_execute "dmp" "${DMP_PATH}/*"

wipe_condition "RES" "blk" "all" &&
wipe_execute "blk" "${BLK_PATH}/*"

wipe_condition "PUB" "new" "all" &&
wipe_execute "new" "${NEW_PATH}/*"

wipe_condition "PUB" "old" "all" &&
wipe_execute "old" "${OLD_PATH}/*"
