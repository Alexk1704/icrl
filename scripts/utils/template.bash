#!/bin/bash

#############################################################################################################################
# This script keeps all settings to customize the execution behavior of the rllib_gazebo project!                           #
#                                                                                                                           #
# You can setup both, the global behavior (COMMON_CONFIG) and local behaviors (<COMPONENT>_CONFIG).                         #
# For each component parameters are defined together wither their corresponding value as hashmap.                           #
# Please provide for each parameter an array containing allowed options, sorted by priority.                                #
# The first entry will be the default/fallback if an invalid value is provided.                                             #
#############################################################################################################################

# keine git komponente, muss man sich selbst drum kümmern (gilt für den container gleichermaßen)
# keine login komponente, welche daten irgendwohin schreibt (scp) oder sich einloggt (ssh) etc.
# keine komponente, die nachrichten versendet, wenn z.b. temperatur zu hoch oder fehler etc.

# COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON
declare -r -a LOCATION_COMMON=(
    "temp"                   # if the project data should be on the machine (sda e.g. /tmp)
    "local"                  # if the project data should be on the network (nfs e.g. /mnt)
)

declare -r -a ACCESS_COMMON=(
    "default"                # use always the default      group for access
    "clusterusers"           # use always the clusterusers group for access
)

declare -r -a PATH_COMMON=(
    "isolated"               # use different paths for each user (user in path)
    "shared"                 # use the same path over multiple users
)

declare -r -a MODE_COMMON=(
    "bash"                   # define if the pipeline is triggered by bash (centralized/single node)
    "slurm"                  # define if the pipeline is triggered by slurm (distributed/multiple nodes)
)
# COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON - COMMON

# ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENV
declare -r -a CURRENT_ENVIRON=(
    "consider"               # only extend   current environment variables if missing
    "ignore"                 # also override current environment variables if existing
)

declare -r -a STORE_ENVIRON=(
    "yes"                    # if the original environment should be stored
    "no"                     # if the original environment should be stored
)

declare -r -a LOCK_ENVIRON=(
    "yes"                    # if the environment should be locked
    "no"                     # if the environment should be locked
)
# ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENVIRON - ENV

# PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PRE
declare -r -a RIGHTS_PREPARE=(
    "none"                   # group has no access to any path (700)
    "read"                   # group can read                    paths (740)
    "write"                  # group can read + write            paths (760)
    "execute"                # group can read + write + execute  paths (770)
)

declare -r -a MOUNT_PREPARE=(
    "original"               # do not alter the mount, if it is not accessable this leads to an error
    "fallback"               # do     alter the mount, if it is not accessable a fallback will be provided
)

declare -r -a PATH_PREPARE=(
    "none"                   # do not alter the path, if it is not accessable this leads to an error
    "string"                 # do     alter the path, if it is not accessable this will append a string
    "counter"                # do     alter the path, if it is not accessable this will append a counter
)

declare -r -a UNITE_PREPARE=(
    "none"                   # do not touch anything of the path variable
    "local"                  # reduce subsequent duplicates in the path variable (!= unique/set)
    "global"                 # reduce arbitrary  duplicates in the path variable (== unique/set)
)
# PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PREPARE - PRE

# BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUI
declare -r -a PERFORM_BUILD=(
    "inplace"                # build the workspace in the dedicated build path
    "detached"               # build the workspace where it is and link to it
)

declare -r -a ARCHIVE_BUILD=(
    "none"                   # workspaces are moved/synced as they are
    ".tar"                   # workspaces are moved/synced as tar file
    ".tar.gz"                # workspaces are moved/synced as compressed tar file
)

declare -r -a LEVEL_BUILD=(
    "package"                # all steps are performed on a package   level
    "workspace"              # all steps are performed on a workspace level
)

declare -r -a PLAN_BUILD=(
    "intend"                 # let colcon decide what to do
    "fresh"                  # everything will be forced to build, not only if necessity
    "lazy"                   # nothing    will be forced to build,     only if necessity
)
# BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUILD - BUI

# CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLO
declare -r -a SNAPSHOT_CLONE=(
    "none"                   # none project data will be snapshotted at all
    "joint"                  # all  project data will be snapshotted once
    "apart"                  # each project data will be snapshotted by its own
)

declare -r -a CONSIDER_CLONE=(
    "none"                   # create always a new version
    "name"                   # create only   a new version, if the name not exists
    "hash"                   # create only   a new version, if the hash not matches
)

declare -r -a ARCHIVE_CLONE=(
    "none"                   # project data are moved/synced as they are
    ".tar"                   # project data are moved/synced as tar file
    ".tar.gz"                # project data are moved/synced as compressed tar file
)
# CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLONE - CLO

# DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY
declare -r -a PERFORM_DEPLOY=(
    "inplace"                # deploying the data in the dedicated source path
    "detached"               # deploying the data where it is and link to it
)

declare -r -a ARCHIVE_DEPLOY=(
    "none"                   # versions are moved/synced as they are
    ".tar"                   # versions are moved/synced as tar file
    ".tar.gz"                # versions are moved/synced as compressed tar file
)

declare -r -a STRATEGY_DEPLOY=(
    "refuse"                 # always replace if there is a existing version
    "verify"                 # only   replace if there is a outdated version
    "accept"                 # never  replace if there is a existing version
)

declare -r -a VERSION_DEPLOY=(
    "none"                   # use the original data of the original path
    "match"                  # use the snapshot which id matches
    "oldest"                 # use the snapshot which is the least recent
    "newest"                 # use the snapshot which is the most recent
)
# DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY - DEPLOY

# CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHE
declare -r -a DETAIL_CHECK=(
    "none"                   # do not perform checks at all
    "shallow"                # do     perform checks only on highlevel
    "deep"                   # do     perform checks also on lowlevel
)

declare -r -a EXTENT_CHECK=(
    "none"                   # do not perform machine checks at all
    "hardware"               # do     perform machine checks regarding hardware
    "software"               # do     perform machine checks regarding software
    "all"                    # do     perform machine checks for hardware and software
)

declare -r -a REPORT_CHECK=(
    "print"                  # only print    the log of each check
    "store"                  # only store    the log of each check
    "both"                   # print + store the log of each check
)
# CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHECK - CHE

# EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXE
declare -r -a PROFILE_EXECUTE=(
    "yes"                    # also   exectue a profiler to gather information
    "no"                     # do not execute a profiler to gather information
)

declare -r -a DEBUG_EXECUTE=(
    "yes"                    # also   exectue a debug node to gather information
    "no"                     # do not execute a debug node to gather information
)

declare -r -a GAZEBO_EXECUTE=(
    "headless"               # execute the gazebo simulator in headless mode
    "gui"                    # execute the gazebo simulator in gui      mode
)

declare -r -a PRIORITY_EXECUTE=(
    "normal"                 # define if the execution should be priorized as a common application
    "realtime"               # define if the execution should be priorized as realtime application (affects sheduling/dispatching)
)
# EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXECUTE - EXE

# BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP
declare -r -a ARCHIVE_BACKUP=(
    "none"                   # results are moved/synced as they are
    ".tar"                   # results are moved/synced as tar file
    ".tar.gz"                # results are moved/synced as compressed tar file
)

declare -r -a VERIFY_BACKUP=(
    "none"                   # do not proceed any     validation of the cloned files
    "lazy"                   # do     proceed a lazy  validation of the cloned files
    "exact"                  # do     proceed a exact validation of the cloned files
)

declare -r -a ACTION_BACKUP=(
    "none"                   # do not have any emergency plan, if an error occurs
    "repeat"                 # repeat the full process three times and wait a random time up to a minute
    "rescue"                 # repeat the full process three times and wait a random time up to a minute + resuce the original data on an persisten drive, if possible
)

declare -r -a RESULTS_BACKUP=(
    "integrated"             # runs are not distinguished by their exit code
    "separated"              # runs are     distinguished by their exit code
)
# BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP - BACKUP

# WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE
declare -r -a TEMP_WIPE=(
    "none"                   # do not remove anything
    "wandb"                  # remove wandb   data between runs
    "ray"                    # remove ray     data between runs
    "log"                    # remove ros/ign data between runs
    "all"                    # remove all     data between runs
)

declare -r -a ORG_WIPE=(
    "none"                   # do not remove anything
    "git"                    # remove git data between runs
    "sif"                    # remove sif data between runs
    "all"                    # remove all data between runs
)

declare -r -a OPT_WIPE=(
    "none"                   # do not remove anything
    "scm"                    # remove scm data between runs
    "var"                    # remove var data between runs
    "all"                    # remove all data between runs
)

declare -r -a TMP_WIPE=(
    "none"                   # do not remove anything
    "bld"                    # remove bld data between runs
    "jnk"                    # remove jnk data between runs
    "all"                    # remove all data between runs
)

declare -r -a WIP_WIPE=(
    "none"                   # do not remove anything
    "src"                    # remove src data between runs
    "dat"                    # remove dat data between runs
    "all"                    # remove all data between runs
)

declare -r -a RES_WIPE=(
    "none"                   # do not remove anything
    "dmp"                    # remove dmp data between runs
    "blk"                    # remove blk data between runs
    "all"                    # remove all data between runs
)

declare -r -a PUB_WIPE=(
    "none"                   # do not remove anything
    "new"                    # renive new data between runs
    "old"                    # renive old data between runs
    "all"                    # renive all data between runs
)
# WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE - WIPE

# AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWA
declare -r -a METHOD_AWAIT=(
    "exit"                   # exit further execution
    "pass"                   # do not wait for experiments (not at all)
    "wait"                   # do     wait for experiments (at least once)
)

declare -r -a ABORT_AWAIT=(
    "never"                  # do not abort the experiment at all
    "change"                 # do     abort the experiment any finishes
    "instant"                # do     abort the experiment immediately
)

declare -r -a HOLD_AWAIT=(
    "none"                   # do not send any signal to exit a run
    "gentle"                 # if the run does not exit itself send sigterm
    "moderate"               # if the run does not exit itself send sigint
    "forceful"               # if the run does not exit itself send sigkill
)

declare -r -a LIMIT_AWAIT=(
    "none"                   # do not limit the trys to abort an experiment
    "single"                 # do     limit the trys to abort an experiment by an timer           (1x300s)
    "multiple"               # do     limit the trys to abort an experiment by an timer + counter (3x300s)
)
# AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWAIT - AWA

# BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE
declare -r -a ARCHIVE_BUNDLE=(
    "none"                   # experiments are moved/synced as they are
    ".tar"                   # experiments are moved/synced as tar file
    ".tar.gz"                # experiments are moved/synced as compressed tar file
)

declare -r -a VERIFY_BUNDLE=(
    "none"                   # do not proceed any     validation of the cloned files
    "lazy"                   # do     proceed a lazy  validation of the cloned files
    "exact"                  # do     proceed a exact validation of the cloned files
)

declare -r -a ACTION_BUNDLE=(
    "none"                   # do not have any emergency plan, if an error occurs
    "repeat"                 # repeat the full process three times and wait a random time up to a minute
    "rescue"                 # repeat the full process three times and wait a random time up to a minute + resuce the original data on an persisten drive, if possible
)

declare -r -a DUPLICATE_BUNDLE=(
    "override"               # experiments are     overriden, if they already exist
    "logrotate"              # experiments are not overriden, if they already exist
)
# BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE - BUNDLE

# RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - REL
declare -r -a RIGHTS_RELEASE=(
    "none"                   # group has no access to any path (700)
    "read"                   # group can read                    paths (740)
    "write"                  # group can read + write            paths (760)
    "execute"                # group can read + write + execute  paths (770)
)

declare -r -a NOTIFY_RELEASE=(
    "none"                   # notify nobody about the final result
    "user"                   # notify user   about the final result
    "group"                  # notify group  about the final result
)

declare -r -a MESSAGE_RELEASE=(
    "yes"                    # if the initiator should     be messaged about the final result
    "no"                     # if the initiator should not be messaged about the final result
)
# RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - RELEASE - REL

# REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REP
declare -r -a SCOPE_REPLACE=(
    "none"                   # move no files at all
    "day"                    # move    files older than last day
    "week"                   # move    files older than last week
    "month"                  # move    files older than last month
    "latest"                 # move    files older than the last n files
)

declare -r -a AGGREGATE_REPLACE=(
    "none"                   # do not aggregate moved files at all
    "time"                   # do     aggregate moved files by time
    "name"                   # do     aggregate moved files by name
)
# REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REPLACE - REP

# CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLE
declare -r -a PROJECT_CLEANUP=(
    "none"                   # do not purge any project files at all
    "partly"                 # purge only temporary project files
    "complete"               # purge all            project files
)

declare -r -a FORCE_CLEANUP=(
    "yes"                    # project files will always be touched
    "no"                     # project files will only   be touched, if they are not further in use
)

declare -r -a ENVIRON_CLEANUP=(
    "none"                   # do not revert any environment changes at all
    "partly"                 # revert only the path variable and the cwd
    "complete"               # revert      the complete environment
)

declare -r -a STATE_CLEANUP=(
    "default"                # try to load and reconstruct the default environment
    "latest"                 # try to load and reconstruct the previous environment (if stored)
)
# CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLEANUP - CLE
