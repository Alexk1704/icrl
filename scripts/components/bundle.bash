#!/bin/bash

#############################################################################################################################
# This component contains the dump logic to collect all experiment data and provide it finally at the PUB location.         #
# Depending on the setup, the collected experiment data will be packed or compressed and stored.                            #
# This will be triggered per experiment and is equivalent to backup, which will be triggered per run.                       #
#############################################################################################################################

# depending on the entry decide what to dump
# dmp of all, latest, oldest or exp_name
#
# if BAK contains only a single folder, use its name for the DMP

# collect all experiments of a group in bak
# and dmp these, if they are ready/or abort them
# wait for them if not ready?!

lookup="$_"
apply_config "${lookup^^}"


if [[ "${BUNDLE_CONFIG['ARCHIVE']}" != "none" ]]
    then print_warn "archive will be created!"
    else print_warn "archive will not be created!"
fi

if [[ "${BUNDLE_CONFIG['VERIFY']}" != "none" ]]
    then print_warn "bundle will be verified!"
    else print_warn "bundle will not be verified!"
fi

if [[ "${BUNDLE_CONFIG['ACTION']}" != "none" ]]
    then print_warn "actions will be triggered!"
    else print_warn "actions will not be triggered!"
fi

if [[ "${BUNDLE_CONFIG['DUPLICATE']}" == "overriden" ]]
    then print_warn "duplicate will be overriden!"
    else print_warn "duplicate will not be overriden!"
fi


# im selben pfad erstellen (auf dem gleichen laufwerk)
# im ziel pfad erstellen (ggf. auf einem anderen laufwerk)
# ist das erste überhaupt sparsammer?!
# exit code von .tar.gz reicht ja
function bundle_archive {
    print_info "create bundle of ${1}"
    case "${BUNDLE_CONFIG['ARCHIVE']}" in
        ".tar")    eval_state $(pack_uncompressed "${2}" "${3}") ;;
        ".tar.gz") eval_state $(pack_compressed   "${2}" "${3}") ;;
    esac
}

function bundle_duplicate {
    if [[ -n $(ls "${DMP_PATH}" | grep "${EXP_NAME}") ]]
        then print_warn "dump already exists in DMP_PATH"
    fi

    print_info "handle duplicate in ${1}"
    case "${BUNDLE_CONFIG['DUPLICATE']}" in
        "override")  annex="" ;;
        "logrotate") annex="_$(ls ${DMP_PATH}/${EXP_NAME}* | wc -l)" ;;
    esac
}

function bundle_move {
    print_info "sync DMP_PATH with BLK_PATH"
    case "${BACKUP_CONFIG['ARCHIVE']}" in
        "none")    eval_state $(save_copy "${1}/"       "${2}" ; sync) ;;
        ".tar")    eval_state $(save_copy "${1}.tar"    "${2}" ; sync) ;;
        ".tar.gz") eval_state $(save_copy "${1}.tar.gz" "${2}" ; sync) ;;
    esac
}

function bundle_hashing {
    print_info "create hashsum of ${2}"
    eval_state $(create_hash "${1}" "${2}")
}

function bundle_compare {
    print_info "compare file ${1} with ${2}"
    eval_state $(compare_file "${1}" "${2}")
}

# was verifizieren?
# archive oder entpackte dateien?
# wo vor dem kopieren, ob archiv passt oder nach dem kopieren ob archive passt
# oder nach dem kopieren, ob die entpackten dateien passen
# rsync stellt ja sicher, dass es passt, sonst exit code != 0
function bundle_verify {
    case "${BUNDLE_CONFIG['VERIFY']}" in
        "full")
            bundle_pack "${1}/source" "${source_path}/*"
            bundle_pack "${1}/target" "${target_path}/*"
            ;;
        "lazy")
            bundle_hashing "${1}/source" "${source_path}"
            bundle_hashing "${1}/target" "${target_path}"
            ;;
    esac

    bundle_compare "${1}/source" "${1}/target"
}

function bundle_action {
    print_info "emergency action for bundle of ${1}"
    case "${BUNDLE_CONFIG['ACTION']}" in
        "repeat") ;;
        "rescue") ;;
    esac
}

#----------------------------------------------------------------------------------------------------------------------------

bundle_archive "DMP_PATH" "${DMP_PATH}/${EXP_NAME}" "${DMP_PATH}/${EXP_NAME}"

bundle_duplicate

bundle_move "${DMP_PATH}/${EXP_NAME}" "${BLK_PATH}"

bundle_verify ||
bundle_action
