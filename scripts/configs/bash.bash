#!/bin/bash

#############################################################################################################################
# This script keeps all settings to customize the execution behavior of the gazebo_sim project!                             #
#                                                                                                                           #
# You can setup both, the global behavior (COMMON_CONFIG) and local behaviors (<COMPONENT>_CONFIG).                         #
# For each component parameters are defined together wither their corresponding value as hashmap.                           #
# Please provide for each parameter an array containing allowed options, sorted by priority.                                #
# The first entry will be the default/fallback if an invalid value is provided.                                             #
#############################################################################################################################

# COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON
declare -A COMMON_CONFIG=(
    ["LOCATION"]="temp"
    ["ACCESS"]="default"
    ["PATH"]="isolated"
    ["MODE"]="bash"
)
# COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON

# ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENV
declare -A ENVIRON_CONFIG=(
    ["CURRENT"]="ignore"
    ["STORE"]="no"
    ["LOCK"]="no"
)
# ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENV

# PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PRE
declare -A PREPARE_CONFIG=(
    ["RIGHTS"]="none"
    ["MOUNT"]="fallback"
    ["PATH"]="none"
    ["UNITE"]="none"
)
# PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PRE

# BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUI
declare -A BUILD_CONFIG=(
    ["PERFORM"]="inplace"
    ["ARCHIVE"]=".tar.gz"
    ["LEVEL"]="package"
    ["PLAN"]="lazy"
)
# BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUI

# CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLO
declare -A CLONE_CONFIG=(
    ["SNAPSHOT"]="apart"
    ["CONSIDER"]="hash"
    ["ARCHIVE"]=".tar.gz"
)
# CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLO

# DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY
declare -A DEPLOY_CONFIG=(
    ["PERFORM"]="inplace"
    ["ARCHIVE"]=".tar.gz"
    ["STRATEGY"]="refuse"
    ["VERSION"]="match"
)
# DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY

# CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHE
declare -A CHECK_CONFIG=(
    ["DETAIL"]="none"
    ["EXTENT"]="none"
    ["REPORT"]="print"
)
# CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHE

# EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXE
declare -A EXECUTE_CONFIG=(
    ["PROFILE"]="no"
    ["DEBUG"]="yes"
    ["GAZEBO"]="gui"
    ["PRIORITY"]="normal"
)
# EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXE

# BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP
declare -A BACKUP_CONFIG=(
    ["ARCHIVE"]=".tar.gz"
    ["VERIFY"]="exact"
    ["ACTION"]="rescue"
    ["RESULTS"]="separated"
)
# BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP

# WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE
declare -A WIPE_CONFIG=(
    ["TEMP"]="ray" # als array oder string array (space separated)
    ["ORG"]="none"
    ["OPT"]="none"
    ["TMP"]="none"
    ["WIP"]="none"
    ["RES"]="none"
    ["PUB"]="none"
)
# WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE

# AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWA
declare -A AWAIT_CONFIG=(
    ["METHOD"]="pass"
    ["ABORT"]="never"
    ["HOLD"]="none"
    ["LIMIT"]="none"
)
# AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWA

# BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE
declare -A BUNDLE_CONFIG=(
    ["ARCHIVE"]=".tar.gz"
    ["VERIFY"]="exact"
    ["ACTION"]="rescue"
    ["DUPLICATE"]="override"
)
# BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE

# RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - REL
declare -A RELEASE_CONFIG=(
    ["RIGHTS"]="none"
    ["NOTIFY"]="none"
    ["MESSAGE"]="no"
)
# RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - REL

# REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REP
declare -A REPLACE_CONFIG=(
    ["SCOPE"]="latest"
    ["AGGREGATE"]="time"
)
# REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REP

# CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLE
declare -A CLEANUP_CONFIG=(
    ["PROJECT"]="partly"
    ["FORCE"]="no"
    ["ENVIRON"]="none"
    ["STATE"]="default"
)
# CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLE
