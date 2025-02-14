#!/bin/bash

#############################################################################################################################
# This component contains
# Depending on the setup
#############################################################################################################################

lookup="$_"
apply_config "${lookup^^}"


if [[ "${REPLACE_CONFIG['SCOPE']}" != "none" ]]
    then print_warn "scope will be latest!"
    else print_warn "scope will not be latest!"
fi

if [[ "${REPLACE_CONFIG['AGGREGATE']}" != "none" ]]
    then print_warn "access will be permitted!"
    else print_warn "access will not be permitted!"
fi


function replace_existence {
    print_info "check existence of ${1}"
    eval_state $(check_existence "${2}")
}

function replace_sync {
    print_info "sync BLK_PATH with NEW_PATH"
    eval_state $(save_copy "${1}" "${2}")
}

function replace_access {
    print_info "check access of ${1}"
    eval_state $(check_access "${2}")
}

function replace_grant {
    print_info "grant access to ${1}"
    eval_state $(modify_access "${2}" "$(id -nu)" "$(id -ng)" "execute" "${REPLACE_CONFIG['RIGHTS']}")
}

function replace_notify {
    print_info "notify ${1} about new replace"
    eval_state $(notify_user "${2}" "${3}" "${EXP_NAME} of ${PROJECT} now available under ${NEW_PATH}!")
}

function replace_runcommands {
    print_info "deposit message for ${1} via sshrc"
    eval_state $(append_file "${2}" "${PROJECT}: ${EXP_NAME} @ ${NEW_PATH}!")
}

#----------------------------------------------------------------------------------------------------------------------------

case "${REPLACE_CONFIG['SCOPE']}" in
    "day") : ;;
    "week") : ;;
    "month") : ;;
    "latest") : ;;
esac

# replace_swap "OLD_PATH" "${NEW_PATH}" "${OLD_PATH}"
