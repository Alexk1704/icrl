#!/bin/bash

#############################################################################################################################
# This component contains the backup logic to secure the experiment data from the WIP location back to the PUB location.    #
# Depending on the setup, the experiment data will be packed or compressed and be stored in some structure.                 #
# This will be triggered per run and is equivalent to dump, which will be triggered per experiment.                         #
#############################################################################################################################

# create tar locally first and then copy with rsync
# versus create tar directly at the destination
# the same applies to the move command

lookup="$_"
apply_config "${lookup^^}"


if [[ "${BACKUP_CONFIG['ARCHIVE']}" != "none" ]]
    then print_warn "archive will be created!"
    else print_warn "archive will not be created!"
fi

if [[ "${BACKUP_CONFIG['VERIFY']}" != "none" ]]
    then print_warn "backup will be verified!"
    else print_warn "backup will not be verified!"
fi

if [[ "${BACKUP_CONFIG['ACTION']}" != "none" ]]
    then print_warn "actions will be triggered!"
    else print_warn "actions will not be triggered!"
fi

if [[ "${BACKUP_CONFIG['RESULTS']}" == "integrate" ]]
    then print_warn "results will be integrated!"
    else print_warn "results will not be integrated!"
fi


function backup_logs {
    print_info "include ${1} logs in EXP_PATH"
    eval_state $(save_copy "${2}" "${3}")
}

# im selben pfad erstellen (auf dem gleichen laufwerk)
# im ziel pfad erstellen (ggf. auf einem anderen laufwerk)
# ist das erste überhaupt sparsammer?!
# exit code von .tar.gz reicht ja
function backup_archive {
    print_info "create backup of ${1}"
    case "${BACKUP_CONFIG['ARCHIVE']}" in
        ".tar")    eval_state $(pack_uncompressed "${2}" "${3}") ;;
        ".tar.gz") eval_state $(pack_compressed   "${2}" "${3}") ;;
    esac
}

function backup_structure {
    print_info "create results structure in ${1}"
    case "${BACKUP_CONFIG['RESULTS']}" in
        "integrated")
            eval_state $(create_directory "${2}")
            ;;
        "separated")
            eval_state $(create_directory "${2}/success")
            eval_state $(create_directory "${2}/failed")
            ;;
    esac
}

function backup_move {
    print_info "sync DAT_PATH with EXP_PATH"
    case "${BACKUP_CONFIG['ARCHIVE']}" in
        "none")    eval_state $(save_copy "${1}/"       "${2}" ; sync) ;;
        ".tar")    eval_state $(save_copy "${1}.tar"    "${2}" ; sync) ;;
        ".tar.gz") eval_state $(save_copy "${1}.tar.gz" "${2}" ; sync) ;;
    esac
}

function backup_sync {
    print_info "sync EXP_PATH with DMP_PATH"
    eval_state $(save_copy "${1}/" "${2}" ; sync)
}

function backup_hashing {
    print_info "create hashsum of ${2}"
    eval_state $(create_hash "${1}" "${2}")
}

function backup_compare {
    print_info "compare file ${1} with ${2}"
    eval_state $(compare_file "${1}" "${2}")
}

# was verifizieren?
# archive oder entpackte dateien?
# wo vor dem kopieren, ob archiv passt oder nach dem kopieren ob archive passt
# oder nach dem kopieren, ob die entpackten dateien passen
# rsync stellt ja sicher, dass es passt, sonst exit code != 0
function backup_verify {
    case "${BACKUP_CONFIG['VERIFY']}" in
        "full")
            backup_pack "${1}/source" "${source_path}/*"
            backup_pack "${1}/target" "${target_path}/*"
            ;;
        "lazy")
            backup_hashing "${1}/source" "${source_path}"
            backup_hashing "${1}/target" "${target_path}"
            ;;
    esac

    backup_compare "${1}/source" "${1}/target"
}

function backup_action {
    print_info "emergency action for backup of ${1}"
    case "${BACKUP_CONFIG['ACTION']}" in
        "repeat") ;;
        "rescue") ;;
    esac
}

#----------------------------------------------------------------------------------------------------------------------------

if (( $EXIT_CODE != 0 ))
    then
        backup_logs "ros" "${HOME}/.ros/" "${DAT_PATH}/${EXP_ID}/ros/"
        backup_logs "gz"  "${HOME}/.gz/"  "${DAT_PATH}/${EXP_ID}/gz/"
fi

backup_archive "DAT_PATH" "${DAT_PATH}/${EXP_ID}" "${DAT_PATH}/${EXP_ID}"

backup_structure "EXP_PATH" "${EXP_PATH}/results"

if (( $EXIT_CODE == 0 ))
    then backup_move "${DAT_PATH}/${EXP_ID}" "$(fallback_path ${EXP_PATH}/results/success)"
    else backup_move "${DAT_PATH}/${EXP_ID}" "$(fallback_path ${EXP_PATH}/results/failed)"
fi

backup_sync "${EXP_PATH}" "${DMP_PATH}/${EXP_NAME}" # HACK

backup_verify ||
backup_action
