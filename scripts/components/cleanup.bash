#!/bin/bash

#############################################################################################################################
# This component contains the cleanup logic to remove unneccessary data and revert environment changes.                     #
# Depending on the setup, the scopre of purging project data and reverting envrionment changes be selected.                 #
# This will be triggered per experiment and is equivalent to wipe, which will be triggered per run.                         #
# However, this component have to be the last one, which is called over the whole pipeline, to retain the shell session.    #
#############################################################################################################################

# this may be an issue, if someone is working in the same directory at the same time
# execute in dependency of shared/isolated and never delete the toplevel directories
# check if file locks or handles exist on some files under the cleanup directory

# execute before and after, if the script was canceled to cleanup before the run?
# but these files are from a different experiment/user
# however, this run can only be conducted, if the node is assigned exclusively!

# cleanup on slurm manager different to cleanup on slurm node, consider both?
# two different pipelines (execution on manager versus node), since the build should be done only once
# the cleanup makes no sense, as each run will be scheduled in its own session

# environment must be resourced each time, as each run is within a new session
# therefore this script is not really meaningful on the cluster...
# would only be relevant if there are several runs in the same session
# so it can be considered, whether the environment is taken or not

# only relevant outside the container?
# generally, if the use will continue to use the same shell

lookup="$_"
apply_config "${lookup^^}"


if [[ "${CLEANUP_CONFIG['PROJECT']}" != "none" ]]
    then print_warn "project files will be purged!"
    else print_warn "project files will not be purged!"
fi

if [[ "${CLEANUP_CONFIG['FORCE']}" == "yes" ]]
    then print_warn "purge will be forced!"
    else print_warn "purge will not be forced!"
fi

if [[ "${CLEANUP_CONFIG['ENVIRON']}" != "none" ]]
    then print_warn "shell environ will be recovered!"
    else print_warn "shell environ will not be recovered!"
fi

if [[ "${CLEANUP_CONFIG['STATE']}" == "default" ]]
    then print_warn "state will be default!"
    else print_warn "state will not be default!"
fi


function cleanup_force {
    :
}

function cleanup_project {
    print_info "purging project files"
    case "${CLEANUP_CONFIG['PROJECT']}" in
        "partly") : ;;
        "complete") : ;;
    esac
}

function cleanup_state {
    # reload stored env
    :
}

function cleanup_environ {
    print_info "retaining shell environ"
    case "${CLEANUP_CONFIG['ENVIRON']}" in
        "partly")
            export PWD="${_PWD}"
            export PATH="${_PATH}"
            ;;
        "complete")
            # exec -c $0
            exec -c bash -li

            # env -i $0
            env -i bash -li

            source ~/.profile
            source ~/.bashrc
            ;;
    esac
}

#----------------------------------------------------------------------------------------------------------------------------

cleanup_project
cleanup_environ
