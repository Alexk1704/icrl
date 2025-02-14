#!/bin/bash

#############################################################################################################################
# This component contains
# Depending on the setup
#############################################################################################################################

lookup="$_"
apply_config "${lookup^^}"


if [[ "${RELEASE_CONFIG['RIGHTS']}" != "none" ]]
    then print_warn "access will be permitted!"
    else print_warn "access will not be permitted!"
fi

if [[ "${RELEASE_CONFIG['NOTIFY']}" == "none" ]]
    then print_warn "notification will be send!"
    else print_warn "notification will not be send!"
fi

if [[ "${RELEASE_CONFIG['MESSAGE']}" == "yes" ]]
    then print_warn "message will be deposit!"
    else print_warn "message will not be deposit!"
fi


function release_existence {
    print_info "check existence of ${1}"
    eval_state $(check_existence "${2}")
}

function release_sync {
    print_info "sync BLK_PATH with NEW_PATH"
    eval_state $(save_copy "${1}" "${2}")
}

function release_access {
    print_info "check access of ${1}"
    eval_state $(check_access "${2}")
}

function release_grant {
    print_info "grant access to ${1}"
    eval_state $(modify_access "${2}" "$(id -nu)" "$(id -ng)" "execute" "${RELEASE_CONFIG['RIGHTS']}")
}

function release_unotify {
    print_info "notify user ${1} about new release"
    eval_state $(notify_user "${2}" "${EXP_NAME} of ${PROJECT} now available under ${NEW_PATH}!")
}

function release_gnotify {
    print_info "notify group ${1} about new release"
    eval_state $(notify_group "${2}" "${EXP_NAME} of ${PROJECT} now available under ${NEW_PATH}!")
}

function release_message {
    print_info "deposit message for ${1} via sshrc"
    eval_state $(append_file "${2}" "${PROJECT}: ${EXP_NAME} @ ${NEW_PATH}!")
}

#----------------------------------------------------------------------------------------------------------------------------

release_sync "${BLK_PATH}/${EXP_NAME}" "${NEW_PATH}/"

release_grant "PUB_PATH" "${PUB_PATH}"

case "${RELEASE_CONFIG['NOTIFY']}" in
    "user")
        for usr in ${!PROJECT_USERS[@]}
        do
            release_unotify "${PROJECT_USERS[${usr}]}" "${usr}"
        done
        ;;
    "group")
        for grp in ${!PROJECT_GROUPS[@]}
        do
            release_gnotify "${PROJECT_GROUPS[${grp}]}" "${grp}"
        done
        ;;
esac

if [[ "${RELEASE_CONFIG['MESSAGE']}" == "yes" ]]
    then release_message "${PROJECT_USERS[${USER}]}" "${HOME}/.ssh/msg"
fi
