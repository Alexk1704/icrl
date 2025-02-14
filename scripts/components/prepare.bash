#!/bin/bash

#############################################################################################################################
# This component contains logic to handle and ensure the main execution-related paths are existent and accessable.          #
# Depending on the setup duplicate entries within a path can be reduced and rights will be set.                             #
# Please consider all local and remote mounts are provided, so an alternative path an be found if it is not accessable.     #
# This can be either be a missing bind of the container, or due to wrong or missing access rights of the executing user.    #
#############################################################################################################################

# sould synced with the isolated/shared setting, but can be set independently
# share already initially access to some temporary paths as WIP and PUB

# same for wipe/cleanup (be aware of other users)
# shared only local machine (single-user system)
# isolated always on cluster (multi-user system)

# assign rights directly when a directory is created?!
# not possible, as the python code will create most of them

lookup="$_"
apply_config "${lookup^^}"


if [[ "${PREPARE_CONFIG['RIGHTS']}" != "none" ]]
    then print_warn "access will be permitted!"
    else print_warn "access will not be permitted!"
fi

if [[ "${PREPARE_CONFIG['MOUNT']}" != "original" ]]
    then print_warn "mounts will be adjusted!"
    else print_warn "mounts will not be adjusted!"
fi

if [[ "${PREPARE_CONFIG['PATH']}" != "none" ]]
    then print_warn "paths will be adjusted!"
    else print_warn "paths will not be adjusted!"
fi

if [[ "${PREPARE_CONFIG['UNITE']}" != "none" ]]
    then print_warn "paths will be merged!"
    else print_warn "paths will not be merged!"
fi


function prepare_existence {
    print_info "check existence of ${1}"
    eval_state $(check_existence "${2}")
}

function prepare_access {
    print_info "check access of ${1}"
    eval_state $(check_access "${2}")
}

function prepare_ensure {
    print_info "ensure directory ${1} exists"
    eval_state $(ensure_directory "${2}")
}

function prepare_grant {
    print_info "grant access to ${1}"
    eval_state $(modify_access "${2}" "$(id -nu)" "$(id -ng)" "execute" "${PREPARE_CONFIG['RIGHTS']}")
}

function prepare_export {
    print_info "change export of ${1}"
    assign_value "${1}" "${2}" ; eval_state
    # eval_state $(export_writable "${1}" "${2}") # <--- this won't work
    # export_writable "${1}" "${2}" ; eval_state    # <--- but this will

    # referencing as with declare -n is also not working
    # since it will refere the subshell variable not the parent one
}

function prepare_substitute {
    print_warn "replace each ${1} by ${2}"
    for line in $(env | grep "${1}")
    do
        var="${line%=*}"
        val="${line#*=}"
        prepare_export "${var}" "${val/${1}/${2}}"
    done
}

function prepare_basepath {
    # do not use the first existing mount, search for matching paths?
    # e.g., if a git folder exist under some mount use this as base_path
    case "${1}" in
        "TEMP_MOUNT")   fallbacks=("LOCAL_MOUNT")  ;;
        "LOCAL_MOUNT")  fallbacks=("TEMP_MOUNT")   ;;
        "HOME_MOUNT")   fallbacks=("REMOTE_MOUNT") ;;
        "REMOTE_MOUNT") fallbacks=("HOME_MOUNT")   ;;
    esac

    path="${2}"
    for base_path in "${fallbacks[@]}"
    do
        print_info "test ${base_path} as candidate for ${1}"
        eval_state $(check_existence "${!base_path}" && check_access "${!base_path}") &&
        path="${!base_path}" ; break
    done

    if [[ "${!1}" != "${path}" ]]
        then
            prepare_export "${1}" "${path}"
            print_warn "altered mount $(declare -p ${1})"
    fi
}

function prepare_actualpath {
    path="${2}"
    for actual_path in $(list_path "${2}")
    do
        print_info "test ${actual_path} as candidate for ${1}"
        eval_state $(check_existence "${actual_path}" && check_access "${actual_path}") &&
        path="${actual_path}" ; break
    done

    case "${PREPARE_CONFIG['PATH']}" in
        "string")
            counter="$(list_path "${2}" | grep -Pi '[\.\w]+' | wc -l)"
            if (( $counter > 0 ))
                then path="${2}.$(printf '.bak%.0s' {0..$counter})"
            fi
            ;;
        "counter")
            counter="$(list_path "${2}" | grep -Pi '[\.\d]+' | wc -l)"
            if (( $counter > 0 ))
                then path="${2}.$counter"
            fi
            ;;
    esac

    if [[ "${!1}" != "${path}" ]]
        then
            prepare_export "${1}" "${path}"
            print_warn "altered path $(declare -p ${1})"
    fi
}

function prepare_deduplicate {
    case "${PREPARE_CONFIG['UNITE']}" in
        "local")  prepare_export "${1}" "$(deduplicate_local  ${2} /)" ;;
        "global") prepare_export "${1}" "$(deduplicate_global ${2} /)" ;;
    esac
}

function prepare_mount {
    print_info "preparing ${1}=${2}"

    if [[ "${PREPARE_CONFIG['MOUNT']}" != "original" ]]
        then
            prepare_existence "${1}" "${2}" &&
            prepare_access    "${1}" "${2}" ||
            prepare_basepath  "${1}" "${2}"
    fi

    # at this point !1 and no longer $2
    # prepare_ensure "${1}" "${!1}"
    # prepare_grant  "${1}" "${!1}"

    if [[ -n "${!1}" ]]
        then
            # prepare_deduplicate "${1}" "${!1}"
            prepare_substitute "${1}" "${!1}"
        else
            print_err "Cannot substitute ${1}"
            exit -1
    fi
}

function prepare_path {
    print_info "preparing ${1}=${2}"

    if [[ "${PREPARE_CONFIG['PATH']}" != "none" ]]
        then
            prepare_existence  "${1}" "${2}" &&
            prepare_access     "${1}" "${2}" ||
            prepare_actualpath "${1}" "${2}"
    fi

    # at this point !1 and no longer $2
    prepare_ensure "${1}" "${!1}"
    # prepare_grant  "${1}" "${!1}"

    if [[ -n "${!1}" ]]
        then
            prepare_deduplicate "${1}" "${!1}"
            prepare_substitute  "${1}" "${!1}"
        else
            print_err "Cannot substitute ${1}"
            exit -1
    fi
}

#----------------------------------------------------------------------------------------------------------------------------

# must not be uniq
prepare_mount "TEMP_MOUNT" "${TEMP_MOUNT}"
prepare_mount "HOME_MOUNT" "${HOME_MOUNT}"

prepare_mount "LOCAL_MOUNT"  "${LOCAL_MOUNT}"
prepare_mount "REMOTE_MOUNT" "${REMOTE_MOUNT}"

# check if uniq
prepare_path "ORG_PATH" "${ORG_PATH}"
prepare_path "SIF_PATH" "${SIF_PATH}"
prepare_path "GIT_PATH" "${GIT_PATH}"

prepare_path "TMP_PATH" "${TMP_PATH}"
prepare_path "BLD_PATH" "${BLD_PATH}"
prepare_path "JNK_PATH" "${JNK_PATH}"

prepare_path "OPT_PATH" "${OPT_PATH}"
prepare_path "SCM_PATH" "${SCM_PATH}"
prepare_path "VAR_PATH" "${VAR_PATH}"

prepare_path "WIP_PATH" "${WIP_PATH}"
prepare_path "SRC_PATH" "${SRC_PATH}"
prepare_path "DAT_PATH" "${DAT_PATH}"

prepare_path "RES_PATH" "${RES_PATH}"
prepare_path "DMP_PATH" "${DMP_PATH}"
prepare_path "BLK_PATH" "${BLK_PATH}"

prepare_path "PUB_PATH" "${PUB_PATH}"
prepare_path "NEW_PATH" "${NEW_PATH}"
prepare_path "OLD_PATH" "${OLD_PATH}"
