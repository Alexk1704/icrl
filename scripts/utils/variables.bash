#!/bin/bash

#############################################################################################################################
# This script contains all essential variable definitions.                                                                  #
# These are sourced before the wrapper script invokes any other component (after functions).                                #
# Basic variables should be defined here, so each component must only contain the logic on an abstract level.               #
# Additionally, redundancies are prevented.                                                                                 #
#############################################################################################################################

# EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT
# must be exported only once
declare -gx COMPONENT_POINTER

declare -gx -A COMPONENT_COUNTS
declare -gx -A COMPONENT_STATES

for component in ${COMPONENTS[@]}
do
    name="$(component_name ${component})"
    COMPONENT_COUNTS["${name}"]=0
    COMPONENT_STATES["${name}"]=0
done
# EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT - EXPORT

# OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVE
declare -r -A RULERS=(
    ["NEUTRAL"]="|"
    ["LEFT"]="<"
    ["RIGHT"]=">"
    ["UP"]="‾"
    ["MIDDLE_SINGLE"]="−"
    ["MIDDLE_DOUBLE"]="="
    ["DOWN"]="_"
)

declare -r -A COLORS=(
    ["NONE"]="\e[0m"
    ["RED"]="\e[1;31m"
    ["GREEN"]="\e[1;32m"
    ["YELLOW"]="\e[1;33m"
    ["BLUE"]="\e[1;34m"
    ["PURPLE"]="\e[1;35m"
    ["CYAN"]="\e[1;36m"
)

declare -r -A SYMBOLS=(
    ["INFO"]="i"
    ["WARN"]="!"
    ["OK"]="✓"
    ["ERR"]="✗"
    ["PART"]="~"
    ["UNK"]="?"
)

declare -r -A SIGNALS=(
    ["SIGHUP"]="1"
    ["SIGINT"]="2"
    ["SIGQUIT"]="3"
    ["SIGABRT"]="6"
    ["SIGKILL"]="9"
    ["SIGTERM"]="15"
)

declare -r -A RIGHTS=(
    ["none"]=""
    ["read"]="r"
    ["write"]="rw"
    ["execute"]="rwx"
)
# OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVERALL - OVE

# FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FUR
# decide if path can be local or must be remote
# check if path is local based on the underyling filesystem of the mountingpoint
# nfs(4) = remote and ext(4) = local
declare -r -a BASE_PATHS=(
    "TEMP"
    "HOME"
    "LOCAL"
    "REMOTE"
)

# value -> if persistent
declare -r -A TEMP_MOUNTS=(
    ["/tmp"]="false"
)

declare -r -A LOCAL_MOUNTS=(
    ["/mnt/local"]="true"
    ["/mnt/storage"]="true"
)

declare -r -A HOME_MOUNTS=(
    ["/clusterhome/${USER}"]="true"
    ["/home/${USER}"]="true"
)

declare -r -A REMOTE_MOUNTS=(
    ["/mnt/remote"]="true"
    ["/mnt/scratch"]="true"
)

#----------------------------------------------------------------------------------------------------------------------------

declare -r -A PROJECT_USERS=(
    ["clusteruser00"]="Alexander Gepperth"
    ["clusteruser01"]="Benedikt Bagus"
    ["clusteruser03"]="Alexander Krawczyk"
    ["clusteruser04"]="Yannick Denker"
)

declare -r -A PROJECT_GROUPS=(
    ["clusterusers"]="whole user group"
)

#----------------------------------------------------------------------------------------------------------------------------

# no nestes structs in bash possible
declare -r -A BUILD_SCHEME=(
    ["base"]="ros_gz_interfaces ros_gz_bridge ros_gz_sim"
    ["devel"]="custom_interfaces gazebo_sim"
)

declare -r -a WORKSPACES=(
    "base"
    "devel"
)

declare -r -a BASE_PACKAGES=(
    "ros_gz_interfaces"
    "ros_gz_bridge"
    "ros_gz_sim"
)

declare -r -a DEVEL_PACKAGES=(
    "custom_interfaces"
    "gazebo_sim"
)
# FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FURTHER - FUR
