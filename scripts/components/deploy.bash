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


if [[ "${DEPLOY_CONFIG['PERFORM']}" == "inplace" ]]
    then print_warn "project will be inplace!"
    else print_warn "project will not be inplace!"
fi

if [[ "${DEPLOY_CONFIG['ARCHIVE']}" != "none" ]]
    then print_warn "archive will be created!"
    else print_warn "archive will not be created!"
fi

if [[ "${DEPLOY_CONFIG['STRATEGY']}" != "verify" ]]
    then print_warn "files will be verified!"
    else print_warn "files will not be verified!"
fi

if [[ "${DEPLOY_CONFIG['VERSION']}" == "none" ]]
    then print_warn "version will be none!"
    else print_warn "version will not be none!"
fi


function deploy_link {
    print_info "link ${1} files from ${2} to ${3}"
    eval_state $(create_link "${2}" "${3}")
}

function deploy_move {
    print_info "move ${1} files from ${2} to ${3}"
    eval_state $(save_copy "${2}" "${3}")
}

function deploy_delete {
    print_info "remove all ${1} files"
    eval_state $(protected_delete "${2}")
}

function deploy_hashing {
    print_info "create hashsum of ${2}"
    eval_state $(create_hash "${1}" "${2}")
}

function deploy_compare {
    print_info "compare file ${1} with ${2}"
    eval_state $(compare_file "${1}" "${2}")
}

function deploy_resolve {
    first_entry "$(find_entry ${SCM_PATH} ${1})"
}

function deploy_archive {
    print_info "create archive of ${1}"
    case "${DEPLOY_CONFIG['ARCHIVE']}" in
        ".tar")    eval_state $(pack_uncompressed "${2}" "${3}") ;;
        ".tar.gz") eval_state $(pack_compressed   "${2}" "${3}") ;;
    esac
}

#----------------------------------------------------------------------------------------------------------------------------

case "${DEPLOY_CONFIG['STRATEGY']}" in
    "refuse") : ;;
    "verify") : ;;
    "accept") : ;;
esac

case "${DEPLOY_CONFIG['VERSION']}" in
    "none")   : ;; # "${SRC_PATH}"
    "match")  : ;; # "${SCM_PATH}"
    "oldest") : ;; # "${SCM_PATH}"
    "newest") : ;; # "${SCM_PATH}"
esac

case "${DEPLOY_CONFIG['PERFORM']}" in
    "inplace")
        # deploy_archive "${SCM_PATH}/${EXP_NAME}" "${SCM_PATH}/${EXP_NAME}"

        # problem, weil nur eine datei darunter hängt und die dann als datei erstellt wird anstatt unter das verzeichnis zu packen
        deploy_move "ROS sandbox" "${SIF_PATH}/${ROS_SANDBOX}/" "${SRC_PATH}/${ROS_SANDBOX}/"

        deploy_move "CRL repository" "${GIT_PATH}/${CRL_REPOSITORY}/" "${SRC_PATH}/${CRL_REPOSITORY}/"
        deploy_move "GMM repository" "${GIT_PATH}/${GMM_REPOSITORY}/" "${SRC_PATH}/${GMM_REPOSITORY}/"
        deploy_move "EXP repository" "${GIT_PATH}/${EXP_REPOSITORY}/" "${SRC_PATH}/${EXP_REPOSITORY}/"

        # deploy_resolve
        # # somehow store the clone ids/assignments and use them here
        # deploy_move "ROS sandbox" "${SCM_PATH}/${ROS_path}/" "${SRC_PATH}/${ROS_SANDBOX}/"
        #
        # deploy_move "CRL repository" "${SCM_PATH}/${CRL_path}/" "${SRC_PATH}/${CRL_REPOSITORY}/"
        # deploy_move "GMM repository" "${SCM_PATH}/${GMM_path}/" "${SRC_PATH}/${GMM_REPOSITORY}/"
        # deploy_move "EXP repository" "${SCM_PATH}/${EXP_path}/" "${SRC_PATH}/${EXP_REPOSITORY}/"

        # das sind die appart fälle
        # joint fehlt noch in beiden varianten

        # joint .tar.gz of apart .tar.gz files?
        # in both cases easier to handle, or?

        # if not only one link/move needed
        # but the overal logic is more complicated
        # deploy_move "project data" "${SCM_PATH}/${path}" "${SRC_PATH}/"
        ;;
    "detached")
        deploy_link "ROS sandbox" "${SIF_PATH}/${ROS_SANDBOX}/" "${SRC_PATH}/${ROS_SANDBOX}"

        deploy_link "CRL repository" "${GIT_PATH}/${CRL_REPOSITORY}/" "${SRC_PATH}/${CRL_REPOSITORY}"
        deploy_link "GMM repository" "${GIT_PATH}/${GMM_REPOSITORY}/" "${SRC_PATH}/${GMM_REPOSITORY}"
        deploy_link "EXP repository" "${GIT_PATH}/${EXP_REPOSITORY}/" "${SRC_PATH}/${EXP_REPOSITORY}"

        # deploy_resolve
        # # only works, if the data is unpacked/uncompressed
        # deploy_link "ROS sandbox" "${SCM_PATH}/${ROS_path}/" "${SRC_PATH}/${ROS_SANDBOX}"
        #
        # deploy_link "CRL repository" "${SCM_PATH}/${CRL_path}/" "${SRC_PATH}/${CRL_REPOSITORY}"
        # deploy_link "GMM repository" "${SCM_PATH}/${GMM_path}/" "${SRC_PATH}/${GMM_REPOSITORY}"
        # deploy_link "EXP repository" "${SCM_PATH}/${EXP_path}/" "${SRC_PATH}/${EXP_REPOSITORY}"

        # deploy_link "project data" "${SCM_PATH}/${path}" "${SRC_PATH}"
        ;;
esac

# für die container eine lösung überlegen, die besser ist als die aktuelle
# eine ordner ebene einfügen, dann wäre alles gleichermaßen handhabbar, wie bei git repos
