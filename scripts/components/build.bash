#!/bin/bash

#############################################################################################################################
# This component contains the build instructions for all involved ros packages over our workspaces.                         #
# Depending on the setup, the build process will be performed at the original location or somwhere else detached (/tmp).    #
# Additionally, the user can force a rebuild on package or workspace level.                                                 #
# However, a detached build may suffer from diverging paths, which should be fixed.                                         #
#############################################################################################################################

# kann ein lokaler build genommen werden?
# versus ein build auf dem manager
# versus ein build auf einer kiste
# versus ein build pro kiste

# versus ein build pro experiment
# versus ein build pro run
# je nach stufe der invocation

# if something superflous is created, delete it afterwards
# including temporary/wip files of the build process
# or is colcon doing already some cleanup?

#------------------------------------------------------------------#
# build vor version, damit nicht neu gebaut werden muss (simpler)  #
# build nach version, damit der snapshot kleiner ist (sparsammer)  #
# durch build/version kann so oder so beides neu getriggert werden #
#------------------------------------------------------------------#

lookup="$_"
apply_config "${lookup^^}"


if [[ "${BUILD_CONFIG['PERFORM']}" == "inplace" ]]
    then print_warn "process will be inplace!"
    else print_warn "process will not be inplace!"
fi

if [[ "${BUILD_CONFIG['ARCHIVE']}" != "none" ]]
    then print_warn "archive will be created!"
    else print_warn "archive will not be created!"
fi

if [[ "${BUILD_CONFIG['LEVEL']}" == "workspace" ]]
    then print_warn "level will be workspace!"
    else print_warn "level will not be workspace!"
fi

if [[ "${BUILD_CONFIG['PLAN']}" == "intend" ]]
    then print_warn "plan will be inteneded!"
    else print_warn "plan will not be inteneded!"
fi


function build_cwd {
    print_info "set ${1} as CWD"
    cd "${1}" ; eval_state
}

function build_archive {
    print_info "create archive of ${1}"
    case "${BUILD_CONFIG['ARCHIVE']}" in
        ".tar")    eval_state $(pack_uncompressed "${2}" "${3}") ;;
        ".tar.gz") eval_state $(pack_compressed   "${2}" "${3}") ;;
    esac
}

function build_move {
    print_info "move ${1} files from ${2} to ${3}"
    eval_state $(save_copy "${2}" "${3}")
}

function build_link {
    print_info "link ${1} files from ${2} to ${3}"
    eval_state $(create_link "${2}" "${3}")
}

function build_purge {
    print_info "purge complete ${1} path"
    eval_state $(protected_delete "${2}")
}

function build_clear {
    case "${BUILD_CONFIG['LEVEL']}" in
        "workspace")
            print_info "clear all files of workspace ${1}"
            eval_state $(colcon clean workspace -y)
            ;;
        "pacakge")
            print_info "clear the files of package ${1}"
            eval_state $(colcon clean packages -y --packages-select "${1}")
            ;;
    esac
}

# function build_proceed {
#     case "${BUILD_CONFIG['LEVEL']}" in
#         "workspace")
#             print_info "rebuilding complete workspace ${1}"
#             eval_state $(colcon build --symlink-install --continue-on-error)
#             ;;
#         "package")
#             print_info "rebuilding solely package ${1}"
#             eval_state $(colcon build --symlink-install --continue-on-error --packages-select "${1}" --allow-overriding "${1}")
#             ;;
#     esac
# }

function build_proceed {
    case "${BUILD_CONFIG['LEVEL']}" in
        "workspace")
            print_info "rebuilding complete workspace ${1}"
            eval_state $(colcon build --continue-on-error)
            ;;
        "package")
            print_info "rebuilding solely package ${1}"
            eval_state $(colcon build --continue-on-error --packages-select "${1}" --allow-overriding "${1}")
            ;;
    esac
}

function build_source {
    case "${BUILD_CONFIG['LEVEL']}" in
        "workspace")
            print_info "resourcing whole workspace ${1}"
            eval_state $(source ./install/setup.bash)
            ;;
        "package")
            print_info "resourcing only package ${1}"
            eval_state $(source ./install/${1}/share/${1}/package.bash)
            ;;
    esac
}

function build_check {
    if [[ -v COLCON_PREFIX_PATH ]]
        then print_warn "some workspaces are already registered"
        else print_info "no workspaces are currently registered"
    fi

    if [[ "${AMENT_PREFIX_PATH}" != "${AMENT_CURRENT_PREFIX}" ]]
        then print_warn "some packages are already existing"
        else print_info "no packages are currently existing"
    fi
}

#----------------------------------------------------------------------------------------------------------------------------

build_check

# für build, clone und deploy
# dateien immer direkt im ziel erstellen

# für backup und bundle
# dateien ggf. erst lokal erstellen und dann ins ziel kopieren
# anschließend nochmals verifizieren, dass alles passt

# is this in any case more performant than copy the files directly?
# consider both cases: a) create tar local or b) create tar remote
# operation overhead on storage vs operation overhead on network

# archive missing
case "${BUILD_CONFIG['PERFORM']}" in
    "inplace")  build_move "BUILD" "${GIT_PATH}/${CRL_REPOSITORY}/workspace/" "${BLD_PATH}/workspace/" ;;
    "detached") build_link "BUILD" "${GIT_PATH}/${CRL_REPOSITORY}/workspace"  "${BLD_PATH}/workspace"  ;;
esac

path="${BLD_PATH}/workspace"
for workspace in "${WORKSPACES[@]}"
do
    declare -n PACKAGES="${workspace^^}_PACKAGES"
    for package in "${PACKAGES[@]}"
    do
        build_cwd "${path}/${workspace}" || break

        case "${BUILD_CONFIG['LEVEL']}" in
            "workspace") entity="${workspace}" ;;
            "package")   entity="${package}"   ;;
        esac

        case "${BUILD_CONFIG['PLAN']}" in
            "intend")                               build_proceed "${entity}" ;;
            "force")  (build_clear  "${entity}") && build_proceed "${entity}" ;;
            "lazy")   (build_source "${entity}") || build_proceed "${entity}" ;;
        esac

        case "${BUILD_CONFIG['LEVEL']}" in
            "workspace") break    ;;
            "package")   continue ;;
        esac

        build_cwd "${OLDPWD}" || continue
    done
done

# archive missing
case "${BUILD_CONFIG['PERFORM']}" in
    "inplace")  build_move "BUILD" "${BLD_PATH}/workspace/" "${GIT_PATH}/${CRL_REPOSITORY}/workspace/" ;;
    # "detached") build_link "BUILD" "${BLD_PATH}/workspace"  "${GIT_PATH}/${CRL_REPOSITORY}/workspace"  ;;
esac

# do this in cleanup?
# build_purge "BUILD" "${BLD_PATH}"

##################################
# symbolic links must be fixed   #
# these are pointing to BLD_PATH #
##################################

# workaround by trigger build on node level with lazy plan
# can this be prevented, if --symlink-install is not used
# makes no sense in the case, the code should be deployed
# only useful for active development...
