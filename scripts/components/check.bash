#!/bin/bash

#############################################################################################################################
# This component contains the a few checks mostly to ensure, that the execution will be flawless.                           #
# Depending on the setup, these are more or less extensive and can be additionally wrote in a log file.                     #
# These checks are almost the only possibility to traceback and reproduce possible errors.                                  #
#############################################################################################################################

# first check if command exist
# like some try/catch constrcut
# skip if command not available

# execude each command only once and hold their outputs in variables
# this will reduce the overhead and increase the performance

lookup="$_"
apply_config "${lookup^^}"


if [[ "${CHECK_CONFIG['DETAIL']}" != "none" ]]
    then print_warn "checks will be performed!"
    else print_warn "checks will not be performed!"
fi

if [[ "${CHECK_CONFIG['EXTENT']}" != "none" ]]
    then print_warn "machine will be included!"
    else print_warn "machine will not be included!"
fi

if [[ "${CHECK_CONFIG['REPORT']}" != "print" ]]
    then print_warn "checks will be stored!"
    else print_warn "checks will not be stored!"
fi


function handle_check {
    # local log_file="${LOG_PATH}/$(hostname)_checks.log"

    case "${CHECK_CONFIG['REPORT']}" in
        "print")
            if [[ "${CHECK_CONFIG['DETAIL']}" == "deep" && -n "${2}" ]]
                then printf -- "${1}\n${2}\n"
                else printf -- "${1}\n"
            fi
            ;;
        "store")
            if [[ "${CHECK_CONFIG['DETAIL']}" == "deep" && -n "${2}" ]]
                then printf -- "${1}\n${2}\n" &> "${LOG_PATH}/$(hostname)_checks.log"
                else printf -- "${1}\n" &> "${LOG_PATH}/$(hostname)_checks.log"
            fi
            ;;
        "both")
            if [[ "${CHECK_CONFIG['DETAIL']}" == "deep" && -n "${2}" ]]
                then printf -- "${1}\n${2}\n" |& tee "${LOG_PATH}/$(hostname)_checks.log"
                else printf -- "${1}\n" |& tee "${LOG_PATH}/$(hostname)_checks.log"
            fi
            ;;
    esac
}

#----------------------------------------------------------------------------------------------------------------------------

if [[ "${CHECK_CONFIG['DETAIL']}" == "none" ]]
    then
        print_info "checking overview..."
        handle_check "$(neofetch)"
        return
fi

print_info "checking shell..."
handle_check "shell: ${SHELL}" "${SHELL} --version"
handle_check "flags: ${FLAGS}"
handle_check "shopt: $(shopt)"

print_info "checking invocation..."
handle_check "caller: ${CALLER_NAME}"
handle_check "callee: ${CALLEE_NAME}"

print_info "checking execution..."
handle_check "nice: $(nice)"
handle_check "ppid: ${PPID}"
handle_check "fd 1: $(readlink /proc/$PPID/fd/1)"
handle_check "fd 2: $(readlink /proc/$PPID/fd/2)"

print_info "checking environ..."
handle_check "$(printenv | wc -l) environment variables" "$(printenv)"
# NOTE: includes no bash variables like HOSTNAME
# use env, environ, compgen -v or declare -p

print_info "checking location..."
handle_check "cwd:  ${PWD}"
handle_check "home: ${HOME}"

print_info "checking host..."
handle_check "host: ${HOSTNAME}"

print_info "checking system..."
handle_check "uptime of this node:  $(uptime)"
handle_check "os and release info:  $(lsb_release -a)"

print_info "checking kernel..."
handle_check "kernel and arch info: $(uname -a)"
handle_check "$(lsmod 2> /dev/null | wc -l) kernel modules loaded" "$(lsmod 2> /dev/null)"

print_info "checking user..."
handle_check "user: ${USER}"
handle_check "active users: $(who)" # is empty inside the container
handle_check "assigned ids: $(id)"

print_info "checking processes..."
handle_check "$(ps aux | wc -l) processes" "$(ps aux)"

print_info "checking filesystem..."
handle_check "$(mount | wc -l) mopunting points" "$(mount)"
handle_check "free space: $(df -h)"
handle_check "used space: $(du -hs / ./ ~/ /tmp ${TEMP_MOUNT} ${LOCAL_MOUNT} ${HOME_MOUNT} ${REMOTE_MOUNT})" # duplikate vermeiden und schauen, ob pfad existiert

print_info "checking files..."
handle_check "$(lslocks 2> /dev/null | wc -l) file locks" "$(lslocks 2> /dev/null)"
handle_check "$(lsof 2> /dev/null | wc -l) open files"    "$(lsof 2> /dev/null)"

print_info "checking nfs traffic..."
handle_check "nfs stat info:    $(nfsstat)"
handle_check "nfs io stat info: $(nfsiostat)"

print_info "checking network..."
handle_check "test connection: $(ping -c 1 -w 1 -q 1.1.1.1)"
handle_check "$(ufw status | wc -l) open ports"        "$(ufw status)"
handle_check "$(netstat -Watupn | wc -l) used sockets" "$(netstat -Watupn)"

print_info "checking load..."
handle_check "gather load:      $(dstat --load --cpu-adv --mem-adv --swap --disk --io --net 1 1)"
handle_check "gather card load: $(nvidia-smi -q | grep -i -A 4 \'usage\|utilization\')"

print_info "checking sensors..."
handle_check "gather sensors     : $(sensors)"
handle_check "gather card sensors: $(nvidia-smi -q | grep -i \'speed\|temp\|power\')"

#----------------------------------------------------------------------------------------------------------------------------

if [[ "${CHECK_CONFIG['EXTENT']}" == "hardware" || "${CHECK_CONFIG['EXTENT']}" == "all" ]]
    then
        print_info "checking hardware..."
        if [[ "${CHECK_CONFIG['DETAIL']}" == "deep" ]]
            then
                handle_check "hardware devices" "$(lshw -short -quiet 2> /dev/null)"
                handle_check "bulk devices"     "$(lsblk -l 2> /dev/null)"
                handle_check "card devices"     "$(nvidia-smi -q)"
        fi

        print_info "found $(lshw -short -quiet 2> /dev/null | wc -l) hardware devices"
        for name in "processor" "memory" "display" "network" "disk" # needs sudo
        do
            match=$(lshw -short -quiet 2> /dev/null | grep -i "${name}" | awk '{print $4}')
            if [[ -n $match ]]
                then print_ok "${name} is ${match}"
                else print_err "${name} not there"
            fi
        done

        print_info "found $(lsblk 2> /dev/null | wc -l) bulk devices"
        for name in "disk"
        do
            match=$(lsblk -l 2> /dev/null | grep -i "${name}" | awk '{print $1, $4}')
            if [[ -n $match ]]
                then print_ok "${name} is ${match}"
                else print_err "${name} not there"
            fi
        done

        # ip command not available inside the container
        print_info "checking interfaces..."
        for name in $(ls /sys/class/net/)
        do
            handle_check "${name} has mac $(cat /sys/class/net/${name}/address)"
            # ip?!
        done

        handle_check "dns info:  $(cat /etc/resolv.conf)"
        handle_check "dhcp info: $(cat /var/lib/dhcp/dhclient.leases)"
fi

#----------------------------------------------------------------------------------------------------------------------------

if [[ "${CHECK_CONFIG['EXTENT']}" == "software" || "${CHECK_CONFIG['EXTENT']}" == "all" ]]
    then
        print_info "checking software..."
        if [[ "${CHECK_CONFIG['DETAIL']}" == "deep" ]]
            then
                handle_check "apt packages" "$(apt list --installed 2> /dev/null)"
                handle_check "pip packages" "$(pip list --verbose 2> /dev/null)"
        fi

        print_info "found $(apt list --installed 2> /dev/null | wc -l) apt packages"
        for name in "nvidia" "cuda" "python" "pip" "ros" "gazebo"
        do
            match=$(apt list --installed 2> /dev/null | grep -i "${name}" | awk -F'[/ ]' '{print length($1), $0}' | sort -n | head -1)
            if [[ -n $match ]]
                then print_ok "${name} found in version $(echo -e ${match} | awk '{print $4}')"
                else print_err "${name} not found"
            fi
        done

        print_info "found $(pip list --verbose 2> /dev/null | wc -l) pip packages"
        for name in "ray" "keras" "tensorflow" "numpy" "matplotlib"
        do
            match=$(pip list --verbose 2> /dev/null | grep -i $name | awk -F'[/ ]' '{print length($1), $0}' | sort -n | head -1)
            if [[ -n $match ]]
                then print_ok "${name} found in version $(echo -e ${match} | awk '{print $3}')"
                else print_err "${name} not found"
            fi
        done

        print_info "checking commands..."
        for name in "python" "ros2" "gz"
        do
            if [[ -n $(command -v "${name}") ]]
                then print_ok "${name} command is available"
                else print_warn "${name} command not avilable"
            fi
        done
fi

#----------------------------------------------------------------------------------------------------------------------------

print_info "checking paths..."
for name in "${TEMP_MOUNT}" "${LOCAL_MOUNT}" "${HOME_MOUNT}" "${REMOTE_MOUNT}"
do
    base_path=$(eval realpath "${name}")
    if [[ -e $base_path ]]
        then
            entries=$(ls "${base_path}" | wc -l)
            if (( $entries == 0 ))
                then
                    print_ok "path ${base_path} exist and is empty"
                else
                    print_ok "path ${base_path} exist and is not empty (contains ${entries} entries)"
                    if [[ "${CHECK_CONFIG['DETAIL']}" == "deep" ]] ; then ls -alh "${base_path}" ; fi
            fi

            print_info "ownership: $(stat -c '%U:%G' ${base_path})"
            print_info "permissions: $(stat -c '%A' ${base_path})"

            print_info "type: $(stat -c '%F' ${base_path})"
            print_info "system: $(stat -f -c '%T' ${base_path}) ($(findmnt $(stat -c '%m' ${base_path}) | tail -1 | awk '{print $3}'))"
            print_info "binded: $(stat -c '%m' ${base_path}) ($(findmnt $(stat -c '%m' ${base_path}) | tail -1 | awk '{print $2}'))"
        else
            print_err "path ${base_path} not exist"
    fi
done

print_info "checking directories..."
for name in "${ORG_PATH}" "${OPT_PATH}" "${TMP_PATH}" "${WIP_PATH}" "${RES_PATH}" "${PUB_PATH}"
do
    base_path=$(eval realpath "${name}")
    if [[ -d $base_path ]]
        then
            entries=$(ls "${base_path}" | wc -l)
            if (( $entries == 0 ))
                then
                    print_ok "directory ${base_path} exist and is empty"
                else
                    print_ok "directory ${base_path} exist and is not empty (contains ${entries} entries)"
                    if [[ "${CHECK_CONFIG['DETAIL']}" == "deep" ]] ; then ls -alh "${base_path}" ; fi
            fi

            if [[ -w $base_path && -r $base_path ]]
                then
                    print_ok "directory ${base_path} accessable"

                    print_info "ownership: $(stat -c '%U:%G' ${base_path})"
                    print_info "permissions: $(stat -c '%A' ${base_path})"

                    print_info "type: $(stat -c '%F' ${base_path})"
                    print_info "system: $(stat -f -c '%T' ${base_path}) ($(findmnt $(stat -c '%m' ${base_path}) | tail -1 | awk '{print $3}'))"
                    print_info "binded: $(stat -c '%m' ${base_path}) ($(findmnt $(stat -c '%m' ${base_path}) | tail -1 | awk '{print $2}'))"
                else
                    print_err "directory ${base_path} not accessable"
            fi
        else
            print_err "directory ${base_path} not exist"
    fi
done

#----------------------------------------------------------------------------------------------------------------------------

print_info "checking build..."
for workspace in $(ls "${WIP_PATH}/${PROJECT}/workspace/")
do
    for package in $(ls "${WIP_PATH}/${PROJECT}/workspace/${workspace}/src/")
    do
        if [[ -n $(ros2 pkg list | grep -i "${package}") ]]
            then print_ok "package ${package} of workspace ${workspace} found"
            else print_err "package ${package} of workspace ${workspace} not found"
        fi
    done
done

print_info "checking invocation..."
if $IS_CLUSTER
    then
        if [[ "${HOSTNAME}" =~ "slurm-server" ]]
            then print_warn "running on slurm server"
            else print_ok "running on slurm node"
        fi
    else
        print_err "running not on slurm cluster"
fi

print_info "checking resolve-order..."
if $IS_CONTAINER
    then
        if [[ -d "${HOME}/.local" ]]
            then
                if [[ -n $(pip list --verbose | grep "${USER}") ]]
                    then print_err "local pacakges interfere"
                    else print_warn "local pacakges may interfere"
                fi
            else
                print_ok "no local packages will interfere"
        fi
fi

print_info "checking containerization..."
if $IS_CONTAINER
    then
        if (( $(ps -aux | grep -i "singularity" | wc -l) > 2 ))
            then print_warn "singularity is still running"
            else print_ok "singularity is now running"
        fi
    else
        if (( $(ps -aux | grep -i "singularity" | wc -l) > 1 ))
            then print_warn "singularity is already running"
            else print_ok "singularity is not running"
        fi
fi

print_info "checking nvidia-smi..."
if $IS_CONTAINER
    then
        if [[ ! -v CUDA_HOME ]]
            then print_warn "container invoced without --nv"
            else print_ok "container invoced with --nv"
        fi
fi

print_info "checking git versions..."
for repo in "${CRL_REPOSITORY}" "${GMM_REPOSITORY}" "${EXP_REPOSITORY}"
do
    git -C "${GIT_PATH}/${repo}" rev-parse --short HEAD
done

print_info "checking sif versions..."
for sbox in "${ROS_SANDBOX}"
do
    singularity inspect "${SIF_PATH}/${sbox}/container.sif"
done

#----------------------------------------------------------------------------------------------------------------------------

# slurm version, aber wie?!
# singularity version

# kein Zugriff nach außerhalb möglich
# sprich die packages vom host können nicht zugegriffen werden
# daher nicht, wie virtualenv, was das vorhandene erweitert
# Zugriff auf Dateien außerhalb des Containers ist ja teils möglich
