#!/bin/bash

#############################################################################################################################
# This component contains the versioning control setting for the container/the source code.                                 #
# Depending on the setup, the execution will be based on the original data or on copys.                                     #
# So far a snapshot is created, this can additionally be verified.                                                          #
# If the execution is based on a snapshot, either the container or the source code can be updated, wihout side effect.      #
#############################################################################################################################

# handle logrotate of copies to prevent override/replacement
# but also reduce redundancies or keep them low at least

# use epxeriment name as name
# => each experiment bekomes its own (fresh) copy

# use hash (sha512sum) as name
# => create hash and check if version already exists as copy

lookup="$_"
apply_config "${lookup^^}"


if [[ "${CLONE_CONFIG['SNAPSHOT']}" == "joint" ]]
    then print_warn "snapshot will be joint!"
    else print_warn "snapshot will not be joint!"
fi

if [[ "${CLONE_CONFIG['CONSIDER']}" != "none" ]]
    then print_warn "existing ones will be considered!"
    else print_warn "existing ones will not be considered!"
fi

if [[ "${CLONE_CONFIG['ARCHIVE']}" != "none" ]]
    then print_warn "archive will be created!"
    else print_warn "archive will not be created!"
fi


function clone_collect {
    print_info "collect data of ${1}"
    eval_state $(save_copy "${2}" "${3}")
}

function clone_archive {
    print_info "create archive of ${1}"
    case "${CLONE_CONFIG['ARCHIVE']}" in
        ".tar")    eval_state $(pack_uncompressed "${2}" "${3}") ;;
        ".tar.gz") eval_state $(pack_compressed   "${2}" "${3}") ;;
    esac
}

function clone_move {
    print_info "move ${1} files from ${2} to ${3}"
    eval_state $(save_copy "${2}" "${3}")
}

function clone_delete {
    print_info "remove all ${1} files"
    eval_state $(protected_delete "${2}")
}

function clone_id {
    case "${CLONE_CONFIG['CONSIDER']}" in
        "name") echo "${EXP_NAME}" ;;
        "hash") echo "$(ls -Ral "${1}" | shasum | cut -c -16)" ;;
    esac
}

#----------------------------------------------------------------------------------------------------------------------------

# regarding the consider cases
# how to handle the case, if the previous execution was none, and this one is .tar.gz
# 1) ignore and make a .tar.gz in addition
# 2) remove the previous and make only a .tar.gz
# 3) make nothing and keep only the previous data

case "${CLONE_CONFIG['SNAPSHOT']}" in
    "joint")
        # id=$(clone_id "${SIF_PATH}/${ROS_SANDBOX} ${GIT_PATH}/{${CRL_REPOSITORY},${GMM_REPOSITORY},${EXP_REPOSITORY}}")

        clone_collect "ROS sandbox" "${SIF_PATH}/${ROS_SANDBOX}/" "${JNK_PATH}/cln/${ROS_SANDBOX}/"

        clone_collect "CRL repository" "${GIT_PATH}/${CRL_REPOSITORY}/" "${JNK_PATH}/cln/${CRL_REPOSITORY}/"
        clone_collect "GMM repository" "${GIT_PATH}/${GMM_REPOSITORY}/" "${JNK_PATH}/cln/${GMM_REPOSITORY}/"
        clone_collect "EXP repository" "${GIT_PATH}/${EXP_REPOSITORY}/" "${JNK_PATH}/cln/${EXP_REPOSITORY}/"

        id=$(clone_id "${JNK_PATH}/cln")

        # use multiple -C components to build archive
        clone_archive "project data" "${JNK_PATH}/cln" "${JNK_PATH}/${id}"

        clone_move "project data" "${JNK_PATH}/${id}" "${SCM_PATH}/joint/"
        ;;
    "apart")
        ROS_id=$(clone_id "${SIF_PATH}/${ROS_SANDBOX}")

        CRL_id=$(clone_id "${GIT_PATH}/${CRL_REPOSITORY}")
        GMM_id=$(clone_id "${GIT_PATH}/${GMM_REPOSITORY}")
        EXP_id=$(clone_id "${GIT_PATH}/${EXP_REPOSITORY}")

        # issue since the container is already a file, not a path like for the others
        clone_archive "ROS sandbox" "${SIF_PATH}/${ROS_SANDBOX}" "${SIF_PATH}/${ROS_id}"

        clone_archive "CRL repository" "${GIT_PATH}/${CRL_REPOSITORY}" "${GIT_PATH}/${CRL_id}"
        clone_archive "GMM repository" "${GIT_PATH}/${GMM_REPOSITORY}" "${GIT_PATH}/${GMM_id}"
        clone_archive "EXP repository" "${GIT_PATH}/${EXP_REPOSITORY}" "${GIT_PATH}/${EXP_id}"

        clone_move "ROS sandbox" "${SIF_PATH}/${ROS_id}" "${SCM_PATH}/apart/${ROS_SANDBOX}/"

        clone_move "CRL repository" "${GIT_PATH}/${CRL_id}" "${SCM_PATH}/apart/${CRL_REPOSITORY}/"
        clone_move "GMM repository" "${GIT_PATH}/${GMM_id}" "${SCM_PATH}/apart/${GMM_REPOSITORY}/"
        clone_move "EXP repository" "${GIT_PATH}/${EXP_id}" "${SCM_PATH}/apart/${EXP_REPOSITORY}/"
        ;;
esac

# do this in cleanup?
# deploy_delete "CLONE" "${JNK_PATH}/cln"

# rsync does interpret the * as string, not as glob
# therefore the path is not found (also an issue with and without / -> repeating path)
