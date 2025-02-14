#!/bin/bash

#############################################################################################################################
# This component contains all essential as well as execution-dependent exports.                                             #
# Depending on the setup, a variable will be overriden (default) or not.                                                    #
# However, it will be ensured, that paths exist at least with a fallback value.                                             #
#############################################################################################################################

# introduce a variable if an execution has already been performed
# also a variable, if the code is finished (as boolean) EXIT_CODE works only for a single run without reset!
# => use e.g. in access or for cleanup

# maybe use the COMPONENT_POINTER instead?
# also count the experiment runs per machine
# better decision, which component must be triggered (per run vs per exp)
# same for e.g. await (run only on server, not on host)

# code / version / history / time / xyz path for source control
# use counter, timestamp, hash or experiment as name

# check if base is local (node/server) or remote (nfs)
# git, sif base
# wip, pub base

# add some additional fallback logic for the WIP as well as PUB path
# in particular, if the shared mode is active
# if the directory is not accessable -> use another one (add, e.g., a counter bash code)
# inside them, the experiment/run name differs, so they will be separated (python code)

lookup="$_"
apply_config "${lookup^^}"


if [[ "${ENVIRON_CONFIG['CURRENT']}" == "ignore" ]]
    then print_warn "environment will be ignored!"
    else print_warn "environment will not be ignored!"
fi

if [[ "${ENVIRON_CONFIG['STORE']}" == "yes" ]]
    then print_warn "environment will be stored!"
    else print_warn "environment will not be stored!"
fi

if [[ "${ENVIRON_CONFIG['LOCK']}" == "yes" ]]
    then print_warn "environment will be protected!"
    else print_warn "environment will not be protected!"
fi


function environ_path {
    check_existence "$(resolve_path ${1})" && echo "${1}"
}

function environ_mount {
    shortest_entry "$(list_mounts | grep -i ${1})" "1"
}

function environ_basepath {
    for mnt in "${@}"
    do
        base_path="$(environ_path ${mnt})"
        if [[ -n "${base_path}" ]]
            then echo "${base_path}" ; return
        fi

        base_path="$(environ_mount ${mnt})"
        if [[ -n "${base_path}" ]]
            then echo "${base_path}" ; return
        fi
    done
}

function environ_lock {
    if [[ "${ENVIRON_CONFIG['LOCK']}" == "yes" ]]
        then export_readable "${1}" "${2}"
        else export_writable "${1}" "${2}"
    fi
}

function environ_export {
    if [[ ! -v "${1}" || ! $(declare -p "${1}") =~ "-x" ]]
        then
            environ_lock "${1}" "${2}"
            print_info "create export: $(declare -p ${1})"
        else
            case "${ENVIRON_CONFIG['CURRENT']}" in
                "consider")
                    print_warn "consider export: $(declare -p ${1})"
                    ;;
                "ignore")
                    print_err "ignore export: $(declare -p ${1})"
                    environ_lock "${1}" "${2}"
                    print_ok "override export: $(declare -p ${1})"
                    ;;
            esac
    fi
}

#----------------------------------------------------------------------------------------------------------------------------

# log_file="~/log/$(hostname)_environ.log"

if [[ "${ENVIRON_CONFIG['STORE']}" == "yes" ]]
    then printenv > "~/log/$(hostname)_environ.log"
    # then printenv > "${LOG_PATH}/environ.log"
    # NOTE: includes no bash variables like HOSTNAME
    # use env, environ, compgen -v or declare -p
fi

# INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INV
environ_export "IS_SOURCED"  "$(if [[ ${CALLER_SOURCE} == ${SHELL} ]] ; then echo true ; else echo false ; fi)"
environ_export "IS_ISOLATED" "$(if [[ -v SINGULARITY_CONTAINER ]]     ; then echo true ; else echo false ; fi)"

environ_export "IS_CLUSTER" "$(if [[ $(hostname -I) =~ 10.32.48. ]] ; then echo true ; else echo false ; fi)"
environ_export "IS_SERVER"  "$(if [[ $(hostname) =~ slurm-server ]] ; then echo true ; else echo false ; fi)"
# INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INVOC - INV

# MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOU
environ_export "TEMP_MOUNT" "$(environ_basepath ${!TEMP_MOUNTS[@]})"
environ_export "HOME_MOUNT" "$(environ_basepath ${!HOME_MOUNTS[@]})"

environ_export "LOCAL_MOUNT"  "$(environ_basepath ${!LOCAL_MOUNTS[@]})" # findet /mnt/storage beim server -> ist dann ebenfalls /mnt/scratch
environ_export "REMOTE_MOUNT" "$(environ_basepath ${!REMOTE_MOUNTS[@]})"
# MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOUNT - MOU

# MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC
environ_export "SCRATCH" "REMOTE_MOUNT" # should be handled like HOME
environ_export "PROJECT" "rllib_gazebo"

environ_export "ROS_SANDBOX" "ros2"

environ_export "CRL_REPOSITORY" "icrl"
environ_export "GMM_REPOSITORY" "sccl"
environ_export "EXP_REPOSITORY" "egge"

case "${COMMON_CONFIG['LOCATION']}" in
    "temp")  base_path="TEMP_MOUNT" ;;
    "local") base_path="LOCAL_MOUNT" ;;
esac

# or the project/shared and project/user
case "${COMMON_CONFIG['PATH']}" in
    "shared")   actual_path="${PROJECT}" ;;
    "isolated") actual_path="${USER}/${PROJECT}" ;;
esac
# MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC - MISC

# PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH
environ_export "ORG_PATH" "HOME_MOUNT"
environ_export "SIF_PATH" "ORG_PATH/sif"
environ_export "GIT_PATH" "ORG_PATH/git"

environ_export "TMP_PATH" "TEMP_MOUNT/${actual_path}"
environ_export "BLD_PATH" "TMP_PATH/bld"
environ_export "JNK_PATH" "TMP_PATH/jnk"

environ_export "OPT_PATH" "HOME_MOUNT/${actual_path}"
environ_export "SCM_PATH" "OPT_PATH/scm"
environ_export "VAR_PATH" "OPT_PATH/var"

environ_export "WIP_PATH" "${base_path}/${actual_path}"
environ_export "SRC_PATH" "WIP_PATH/src"
environ_export "DAT_PATH" "WIP_PATH/dat"

environ_export "RES_PATH" "REMOTE_MOUNT/${actual_path}"
environ_export "DMP_PATH" "RES_PATH/dmp"
environ_export "BLK_PATH" "RES_PATH/blk"

environ_export "PUB_PATH" "REMOTE_MOUNT/${actual_path}"
environ_export "NEW_PATH" "PUB_PATH/new"
environ_export "OLD_PATH" "PUB_PATH/old"
# PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH - PATH

# EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP
if [[ -v EGGE ]]
    then print_ok "EGGE found!"
    else print_err "EGGE not found!"
fi

if [[ ! -v EGGE ]]
    then
        environ_export "EXP_STR"   "default"
        environ_export "EXP_TIME"  "$(date +'%y-%m-%d-%H-%M-%S')"
        environ_export "EXP_LABEL" "E0-W0-C0-R0"

        environ_export "EXP_NAME" "${EXP_STR}__${EXP_TIME}"
        environ_export "EXP_ID"   "${EXP_STR}__${EXP_LABEL}"
        environ_export "EXP_PATH" "${GIT_PATH}/${EXP_REPOSITORY}/generated/${EXP_NAME}"
fi
# EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP - EXP

# PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON
environ_export "PYTHONPATH" "${PYTHONPATH:+${PYTHONPATH}:}${SRC_PATH}/sccl/src"
# PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON - PYTHON

# COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON
environ_export "_colcon_cd_root" "/opt/ros/rolling"

source /usr/share/colcon_cd/function/colcon_cd.sh
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
# COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON - COLCON

# ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS
source /opt/ros/rolling/setup.bash
print_info "source ROS_UNDERLAY -> ${AMENT_PREFIX_PATH}"

environ_export "ROS_DOMAIN_ID"                 "42"
environ_export "ROS_AUTOMATIC_DISCOVERY_RANGE" "LOCALHOST"
environ_export "RMW_IMPLEMENTATION"            "rmw_fastrtps_cpp"
# ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS

# GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ
environ_export "GZ_VERSION"           "8"
environ_export "GZ_DISTRO"            "harmonic"
environ_export "GZ_IP"                "127.0.0.1"
environ_export "GZ_PARTITION"         "$(hostname)"
environ_export "GZ_SIM_RESOURCE_PATH" "${GZ_SIM_RESOURCE_PATH:+${GZ_SIM_RESOURCE_PATH}:}${SRC_PATH}/icrl/models"
# GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ

# RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY
environ_export "TUNE_TEMP_DIR"                      "/tmp/ray"
environ_export "TUNE_RESULT_DIR"                    "/tmp/ray_results"
environ_export "TUNE_DISABLE_AUTO_CALLBACK_LOGGERS" "1"
# RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY - RAY
