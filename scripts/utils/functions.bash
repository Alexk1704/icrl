#!/bin/bash

#############################################################################################################################
# This script contains all essential function definitions.                                                                  #
# These are sourced before the wrapper script invokes any other component (before variables).                               #
# Basic functions should be defined here, so each component must only contain the logic on an abstract level.               #
# Additionally, redundancies are prevented.                                                                                 #
#############################################################################################################################

function component_name {
    echo "${1}" | rev | cut -d '/' -f1 | rev | cut -d '.' -f1 | tr '[:lower:]' '[:upper:]'
}

#----------------------------------------------------------------------------------------------------------------------------

function enter_component {
    name="$(component_name ${1})"
    COMPONENT_POINTER="> ${name}"
    (( COMPONENT_COUNTS["${name}"]++ ))
}

function exit_component {
    name="$(component_name ${1})"
    COMPONENT_POINTER="${name} <"
    COMPONENT_STATES["${name}"]=$?
}

#----------------------------------------------------------------------------------------------------------------------------

function print_box {
    echo "${RULERS[${1}]}$(printf "${RULERS[${3}]}%.0s" {1..78})${RULERS[${2}]}"
    echo "${RULERS[${1}]}$(printf " %-76s " "${5}")${RULERS[${2}]}"
    echo "${RULERS[${1}]}$(printf "${RULERS[${4}]}%.0s" {1..78})${RULERS[${2}]}"
}

function print_header { echo ; print_box "NEUTRAL" "NEUTRAL" "UP" "DOWN" "STARTED WITH COMPONENT $(component_name ${1})" ; }
function print_footer { print_box "NEUTRAL" "NEUTRAL" "UP" "DOWN" "FINISHED WITH COMPONENT $(component_name ${1})" ; echo ; }

function print_message {
    echo -e "  [${COLORS[${1}]}${SYMBOLS[${2}]}${COLORS['NONE']}] ${3}"
}

function print_info { print_message "BLUE"   "INFO" "${*}" ; }
function print_warn { print_message "YELLOW" "WARN" "${*}" ; }
function print_ok   { print_message "GREEN"  "OK"   "${*}" ; }
function print_err  { print_message "RED"    "ERR"  "${*}" ; }
function print_part { print_message "CYAN"   "PART" "${*}" ; }
function print_unk  { print_message "PURPLE" "UNK"  "${*}" ; }

#----------------------------------------------------------------------------------------------------------------------------

function check_declaration {
    flag=false

    if [[ $(declare -p "${1}") =~ "declare"    && "${2}" =~ ^"v" ]] ; then flag=true ; fi
    if [[ $(declare -p "${1}") =~ "declare -a" && "${2}" =~ ^"a" ]] ; then flag=true ; fi
    if [[ $(declare -p "${1}") =~ "declare -A" && "${2}" =~ ^"h" ]] ; then flag=true ; fi

    if ! $flag ; then echo "Reference ${1} is not of type ${2}..." ; fi
}

function check_variable { check_declaration "${1}" "v" ; }
function check_array    { check_declaration "${1}" "a" ; }
function check_hashmap  { check_declaration "${1}" "h" ; }

function create_declaration {
    flag=false

    if [[ "${2}" =~ ^"v" ]] ; then declare -g    "${1}" && flag=true ; fi
    if [[ "${2}" =~ ^"a" ]] ; then declare -g -a "${1}" && flag=true ; fi
    if [[ "${2}" =~ ^"h" ]] ; then declare -g -A "${1}" && flag=true ; fi

    if ! $flag ; then echo "Reference ${1} of type ${2} not created..." ; fi
}

function create_variable { create_declaration "${1}" "v" ; }
function create_array    { create_declaration "${1}" "a" ; }
function create_hashmap  { create_declaration "${1}" "h" ; }

function ensure_reference {
    if [[ -v "${!1}" ]]
        then check_declaration  "${1}" "${2}"
        else create_declaration "${1}" "${2}"
    fi
}

function ensure_variable { ensure_reference "${1}" "v" ; }
function ensure_array    { ensure_reference "${1}" "a" ; }
function ensure_hashmap  { ensure_reference "${1}" "h" ; }

#----------------------------------------------------------------------------------------------------------------------------

function check_path {
    flag=false

    if [[ -d "${1}" && "${2}" =~ ^"d" ]] ; then flag=true ; fi
    if [[ -f "${1}" && "${2}" =~ ^"f" ]] ; then flag=true ; fi

    if ! $flag ; then echo "Path ${1} is not of type ${2}..." ; fi
}

function check_directory { check_path "${1}" "d" ; }
function check_file      { check_path "${1}" "f" ; }

function create_path {
    flag=false

    if [[ "${2}" =~ ^"d" ]] ; then mkdir -p "${1}" && flag=true ; fi
    if [[ "${2}" =~ ^"f" ]] ; then touch -p "${1}" && flag=true ; fi

    if ! $flag ; then echo "Path ${1} of type ${2} not created..." ; fi
}

function create_directory { create_path "${1}" "d" ; }
function create_file      { create_path "${1}" "f" ; }

function ensure_path {
    if [[ -e "${1}" ]]
        then check_path "${1}" "${2}"
        else create_path "${1}" "${2}"
    fi
}

function ensure_directory { ensure_path "${1}" "d" ; }
function ensure_file      { ensure_path "${1}" "f" ; }

#----------------------------------------------------------------------------------------------------------------------------

function export_writable {
    if [[ -z "${2}" ]]
        then declare -gx "${1}"
        else declare -gx "${1}"="${2}"
    fi
}

function export_readable {
    if [[ -z "${2}" ]]
        then declare -gxr "${1}"
        else declare -gxr "${1}"="${2}"
    fi
}

#----------------------------------------------------------------------------------------------------------------------------

function assign_value {
    ensure_variable "${1}"
    declare -n REFERENCE="${1}"

    REFERENCE="${2}"
}

function reset_value {
    ensure_variable "${1}"
    declare -n REFERENCE="${1}"

    REFERENCE=""
}

#----------------------------------------------------------------------------------------------------------------------------

function pack_variables {
    ensure_reference "${1}" "hashmap"
    declare -n REFERENCE="${1}"

    for variable in ${!REFERENCE[@]}
    do
        ensure_reference "${variable}" "variable"
        declare -n _REFERENCE="${1}"

        _REFERENCE="${REFERENCE[${variable}]}"
    done
}

function unpack_variables {
    ensure_reference "${1}" "hashmap"
    declare -n REFERENCE="${1}"

    for variable in ${@:2}
    do
        ensure_reference "${variable}" "variable"
        declare -n _REFERENCE="${1}"

        REFERENCE["${variable}"]="${_REFERENCE}"
    done
}

function apply_config {
    name="$(component_name ${1})"

    main_config="${name}_CONFIG"

    ensure_reference "${main_config}" "hashmap"
    declare -n REFERENCE="${main_config}"

    echo "Parsing ${#REFERENCE[@]} entries of ${main_config}..."

    for entry in ${!REFERENCE[@]}
    do
        sub_config="${entry}_${name}"

        ensure_reference "${sub_config}" "array"
        declare -n _REFERENCE="${sub_config}"

        echo "Loading ${sub_config}=${REFERENCE[${entry}]}..."

        if [[ ! " ${_REFERENCE[@],,} " =~ " ${REFERENCE[${entry}],,} " ]]
            then
                echo "Illegal value <${REFERENCE[${entry}]}>, loading fallback >${_REFERENCE[0]}<!"
                REFERENCE[${entry}]=${_REFERENCE[0]}
        fi
    done
}

#----------------------------------------------------------------------------------------------------------------------------

function first_entry {
    echo -e "${1}" | awk -v k=${2} '{print $k}' | head -n 1
}

function last_entry {
    echo -e "${1}" | awk -v k=${2} '{print $k}' | tail -n 1
}

function longest_entry {
    echo -e "${1}" | awk -v k=${2} '{print length($k), $k}' | sort -n -k1 | cut -d' ' -f2 | head -n 1
}

function shortest_entry {
    echo -e "${1}" | awk -v k=${2} '{print length($k), $k}' | sort -n -k1 | cut -d' ' -f2 | tail -n 1
}

#----------------------------------------------------------------------------------------------------------------------------

function filter_matches {
    ensure_reference "${1}" "${2}"
    declare -n REFERENCE="${1}"

    local MATCHES=()
    for entry in ${REFERENCE[@]}
    do
        if [[ "${entry}" =~ "${3}" ]]
            then MATCHES+=("${entry}")
        fi
    done

    echo ${MATCHES[@]}
}

function check_unique {
    ensure_reference "${1}" "${2}"
    declare -n REFERENCE="${1}"

    local ARR_1=$(echo "${REFERENCE[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')
    local ARR_2=$(echo "${REFERENCE[@]}" | tr ' ' '\n' | sort | uniq | tr '\n' ' ')
    if [[ "${ARR_1[*]}" == "${ARR_2[*]}" ]]
        then echo true
        else echo false
    fi
}

function check_length {
    ensure_reference "${1}" "${2}"
    declare -n REFERENCE="${1}"

    if [[ "${#REFERENCE[@]}" == "${3}" ]]
        then echo true
        else echo false
    fi
}

#----------------------------------------------------------------------------------------------------------------------------

function trim_string {
    local string="${1}"
    local delimiter="${2}"
    local mode="${3}"

    if [[ -z "${mode}" || "${mode}" =~ "^" ]]
        then string="${string##*(${delimiter})}"
    fi

    if [[ -z "${mode}" || "${mode}" =~ "$" ]]
        then string="${string%%*(${delimiter})}"
    fi

    echo "${string}"
}

function trim_head { trim_string "${1}" "${2}" "^"  ; }
function trim_tail { trim_string "${1}" "${2}" "$"  ; }
function trim_both { trim_string "${1}" "${2}" "^$" ; }

function deduplicate_string {
    local string="${1}"
    local delimiter="${2}"
    local mode="${3}"

    if [[ "${mode}" =~ ^"g" ]]
        then string=$(echo -n "${string}" | tr "${delimiter}" "\n" | awk '!seen[$0]++' | tr "\n" "${delimiter}")
    fi

    if [[ "${mode}" =~ ^"l" ]]
        then string=$(echo -n "${string}" | tr "${delimiter}" "\n" | uniq | tr "\n" "${delimiter}")
    fi

    trim_both "${string}" "${delimiter}"
}

function deduplicate_global { deduplicate_string "${1}" "${2}" "g" ; }
function deduplicate_local  { deduplicate_string "${1}" "${2}" "l" ; }

function validate_path {
    local string="${1}"
    local delimiter="${2}"

    # IFS=\'${2}\'
    for entry in $(echo "${string}" | tr \'${delimiter}\' '\n')
    do
        # $(eval realpath "${path}")
        if [[ ! -e "${entry}" ]]
            then return -1
        fi
    done
}

#----------------------------------------------------------------------------------------------------------------------------

function check_parameter {
    local parameter="${1}"

    if [[ -z "${parameter}" ]]
        then return -1
    fi
}

function check_existence {
    local path="${1}"

    check_parameter "${path}" || return -1

    # $(eval realpath "${path}")
    if [[ ! -e "${path}" ]]
        then return -1
    fi
}

function compare_path {
    local path_1="${1}"
    local path_2="${2}"

    check_parameter "${path_1}" || return -1
    check_parameter "${path_2}" || return -1

    if [[ $(eval realpath "${path_1}") != $(eval realpath "${path_2}") ]]
        then return -1
    fi
}

function compare_directory {
    local directory_1="${1}"
    local directory_2="${2}"

    check_parameter "${directory_1}" || return -1
    check_parameter "${directory_2}" || return -1

    if ! $(diff -r "${directory_1}" "${directory_2}" > /dev/null)
        then return -1
    fi
}

function compare_file {
    local file_1="${1}"
    local file_2="${2}"

    check_parameter "${file_1}" || return -1
    check_parameter "${file_2}" || return -1

    if ! $(cmp "${file_1}" "${file_2}" > /dev/null)
        then return -1
    fi
}

function check_protected {
    local path="${1}"

    check_parameter "${path}" || return -1

    for base_path in ${BASE_PATHS[@]}
    do
        if [[ $(resolve_path "${!base_path}/*") =~ $(resolve_path "${1}") ]]
            then return 0
        fi
    done

    return -1
}

function check_nfs {
    local path="${1}"

    check_parameter "${path}" || return -1

    if [[ ! $(findmnt -n -T "${path}") =~ "nfs" ]]
        then return -1
    fi
}

function check_space {
    local source="${1}"
    local target="${2}"

    check_parameter "${source}" || return -1
    check_parameter "${target}" || return -1

    if (( $(df $(fallback_path "${target}") | tail -n 1 | awk '{print $4}') - $(du -s $(fallback_path "${source}") | tail -n 1 | awk '{print $1}') <= 0 ))
        then return -1
    fi
}

function protected_delete {
    local path="${1}"

    check_parameter "${path}" || return -1

    check_protected "${path}" || return -1
    rm -rf "${path}"
}

function save_copy {
    local source="${1}"
    local target="${2}"

    check_parameter "${source}" || return -1
    check_parameter "${target}" || return -1

    check_space "${source}" "${target}" || return -1
    rsync -a --mkpath "${source}"* "${target}"
}

function check_size {
    local path="${1}"

    check_parameter "${path}" || return -1

    if [[ -s "${path}" ]]
        then return -1
    fi
}

function check_owner {
    local path="${1}"

    check_parameter "${path}" || return -1

    if [[ ! -O "${path}" ]]
        then return -1
    fi
}

function check_group {
    local path="${1}"

    check_parameter "${path}" || return -1

    if [[ ! -G "${path}" ]]
        then return -1
    fi
}

function check_permissions {
    local path="${1}"

    check_parameter "${path}" || return -1

    if ! $(check_owner "${path}") && ! $(check_group "${path}")
        then return -1
    fi
}

function modify_permissions {
    local path="${1}"
    local owner="${2}"
    local group="${3}"

    check_parameter "${path}" || return -1
    check_parameter "${owner}" || return -1
    check_parameter "${group}" || return -1

    check_permissions "${path}" || return -1
    chown -R "${owner}" "${path}"
    chgrp -R "${group}" "${path}"
}

function check_read {
    local path="${1}"

    check_parameter "${path}" || return -1

    if [[ ! -r "${path}" ]]
        then return -1
    fi
}

function check_write {
    local path="${1}"

    check_parameter "${path}" || return -1

    if [[ ! -w "${path}" ]]
        then return -1
    fi
}

function check_privileges {
    local path="${1}"

    check_parameter "${path}" || return -1

    if ! $(check_read "${path}") && ! $(check_write "${path}")
        then return -1
    fi
}

function modify_privileges {
    local path="${1}"
    local urights="${2}"
    local grights="${3}"

    check_parameter "${path}" || return -1
    check_parameter "${urights}" || return -1
    check_parameter "${grights}" || return -1

    check_permissions "${path}" || return -1
    chmod -R "u=${RIGHTS[${urights}]}" "${path}"
    chmod -R "g=${RIGHTS[${grights}]}" "${path}"
}

function check_access {
    local path="${1}"

    check_parameter "${path}" || return -1

    if ! $(check_permissions "${path}") && ! $(check_privileges "${path}")
        then return -1
    fi
}

function modify_access {
    local path="${1}"
    local owner="${2}"
    local group="${3}"
    local urights="${4}"
    local grights="${5}"

    check_parameter "${path}" || return -1
    check_parameter "${owner}" || return -1
    check_parameter "${group}" || return -1
    check_parameter "${urights}" || return -1
    check_parameter "${grights}" || return -1

    modify_permissions "${path}" "${owner}" "${group}"
    modify_privileges "${path}" "${urights}" "${grights}"
}

#----------------------------------------------------------------------------------------------------------------------------

function compress_path {
    local path="${1}"
    local mode="${2}"

    check_parameter "${path}" || return -1
    check_parameter "${mode}" || return -1

    if [[ "${mode}" =~ ^"f" ]] ; then gzip -f "${path}" ; fi
    if [[ "${mode}" =~ ^"d" ]] ; then gzip -fR "${path}" ; fi
}

function compress_file { compress_path "${1}" "f" ; }
function compress_directory { compress_path "${1}" "d" ; }

function decompress_path {
    local path="${1}"
    local mode="${2}"

    check_parameter "${path}" || return -1
    check_parameter "${mode}" || return -1

    if [[ "${mode}" =~ ^"f" ]] ; then gunzip -f "${path}" ; fi
    if [[ "${mode}" =~ ^"d" ]] ; then gunzip -fR "${path}" ; fi
}

function decompress_file { decompress_path "${1}" "f" ; }
function decompress_directory { decompress_path "${1}" "d" ; }

function pack_archive {
    local directory="${1}"
    local archive="${2}"
    local compress="${3}"

    check_parameter "${directory}" || return -1
    check_parameter "${archive}" || return -1
    check_parameter "${compress}" || return -1

    if [[ "${compress}" == "false" ]]
        then tar -cf "${archive}.tar" -C "${directory}" .
        else tar -czf "${archive}.tar.gz" -C "${directory}" .
    fi
}

function pack_compressed { pack_archive "${1}" "${2}" "true"; }
function pack_uncompressed { pack_archive "${1}" "${2}" "false"; }

function unpack_archive {
    local directory="${1}"
    local archive="${2}"
    local compress="${3}"

    check_parameter "${directory}" || return -1
    check_parameter "${archive}" || return -1
    check_parameter "${compress}" || return -1

    if [[ "${compress}" == "false" ]]
        then tar -xf "${archive}.tar" -C "${directory}" .
        else tar -xzf "${archive}.tar.gz" -C "${directory}" .
    fi
}

function unpack_compressed { unpack_archive "${1}" "${2}" "true"; }
function unpack_uncompressed { unpack_archive "${1}" "${2}" "false"; }

#----------------------------------------------------------------------------------------------------------------------------

function resolve_link {
    local path="${1}"

    check_parameter "${path}" || return -1

    eval realpath "${path}"
}

function resolve_path {
    local path="${1}"

    check_parameter "${path}" || return -1

    eval realpath -s "${path}"
}

function create_link {
    local path="${1}"
    local link="${2}"

    check_parameter "${path}" || return -1
    check_parameter "${link}" || return -1

    ln -sf "${path}" "${link}"
}

#----------------------------------------------------------------------------------------------------------------------------

function hash_file {
    local file="${1}"
    local path="${2}"

    check_parameter "${file}" || return -1
    check_parameter "${path}" || return -1

    cat "${path}" | shasum > "${file}"
}

function hash_directory {
    local file="${1}"
    local path="${2}"

    check_parameter "${file}" || return -1
    check_parameter "${path}" || return -1

    ls -Ral "${path}" | shasum > "${file}"
}

function create_hash {
    local file="${1}"
    local path="${2}"

    check_parameter "${file}" || return -1
    check_parameter "${path}" || return -1

    if [[ -f "${path}" ]] ; then hash_file "${file}" "${path}" ; fi
    if [[ -d "${path}" ]] ; then hash_directory "${file}" "${path}" ; fi
}

#----------------------------------------------------------------------------------------------------------------------------

function create_timestamp {
    date --iso-8601=seconds
}

function list_path {
    ls -d "$(dirname ${1})/*" | grep "$(basename ${1})"
    # find "$(dirname ${1})" -name "$(basename ${1}).*" -maxdepth 1
    # ls only returns the base_name, without dir_name by default
    # find always returns the fully quantified path
}

function list_mounts {
    findmnt --list
    # | grep -i "home" | awk '{print $1}' | grep -E '^/[^/]+/[^/]+$'
    # filter ext4/nfs4?
}

function check_process {
    pgrep -f "${1}"
}

function kill_process {
    pkill -9 -f "${1}"
}

# check if command available (checks)
# check if function exists (einfach so)
# which, whereis, type, compgen, command
# simply use a subshell and evaluate the return value

#----------------------------------------------------------------------------------------------------------------------------

# compare_and
# compare_or
# compare_not

# compare_equal
# compare_unequal

# compare_exact
# compare_contains

# variable equals (all type comparions)
# value
# array (values/+order)
# hashmap (values/+keys)

# parameter einzeln übergeben
# versus als array übergeben
function match_any {
    for keyword in "${@:2}"
    do
        if [[ "${1}" == "${keyword}" ]]
            then return 0
        fi
    done

    return -1
}

function match_all {
    for keyword in "${@:2}"
    do
        if [[ "${1}" != "${keyword}" ]]
            then return -1
        fi
    done

    return 0
}

function contain_entry {
    ensure_reference "${1}" "${2}"
    declare -n REFERENCE="${1}"

    if [[ " ${REFERENCE[@]} " =~ " ${3} " ]]
        then return 0
        else return -1
    fi
}

# parameter einzeln übergeben
# versus zweites array übergeben
function contain_any {
    ensure_reference "${1}" "${2}"
    declare -n REFERENCE="${1}"

    for keyword in "${@:3}"
    do
        if [[ " ${REFERENCE[@]} " =~ " ${keyword} " ]]
            then return 0
        fi
    done

    return -1
}

function contain_all {
    ensure_reference "${1}" "${2}"
    declare -n REFERENCE="${1}"

    for keyword in "${@:3}"
    do
        if [[ ! " ${REFERENCE[@]} " =~ " ${keyword} " ]]
            then return -1
        fi
    done

    return 0
}

#----------------------------------------------------------------------------------------------------------------------------

function find_entry {
    if [[ -z "${3}" ]] ; then find "${1}" -name "${2}" ; fi
    if [[ "${3}" =~ ^"f" ]] ; then find "${1}" -name "${2}" -type "f" ; fi
    if [[ "${3}" =~ ^"d" ]] ; then find "${1}" -name "${2}" -type "d" ; fi
}

function find_file { find_entry "${1}" "${2}" "f"; }
function find_directory { find_entry "${1}" "${2}" "d"; }

function notify_user {
    for pty in $(who | grep "${1}" | awk '{print $2}')
    do
        write "${1}" "${pty}" "${2}"
    done
}

function notify_group {
    wall -g "${1}" "${2}"
}

function append_file {
    mkdir -p $(dirname "${1}") # not clean
    echo "${2}" >> "${1}"
}

# fallback in pfad level, bis existierenden gefunden...
function fallback_path {
    # recursive calling to prevent looping
    # but it must be a even simpler more efficient solution
    check_existence "${1}" && echo $(realpath "${1}") || fallback_path $(dirname "${1}")
}

# + größter/längster gemeinsamer nenner finden...
function common_path {
    # stellt nicht sicher, dass von vorne gematcht wird, bis unequal
    # schaut nur, ob auf der gleichen ebene/tiefe die sachen gleich heißen
    comm -1 -2 <(echo "${1}" | tr "${3}" "\n") <(echo "${2}" | tr "${3}" "\n") | tr '\n' '/'
}

# shortest/longest match
# vorne/hinten angefangen

# dynamisches match, um zu schauen, ob die struktur ab irgendwo gleich ist
# einfach andere basepaths -> von hinten anfangen zu matchen und schauen, was jeweils der rest ist
# was oben im fall mit dem rest machen?

#----------------------------------------------------------------------------------------------------------------------------

# call with eval_command "xyz"
function eval_command {
    eval "${2}" \
    && state=$? ; print_ok "success" ; return $state \
    || state=$? ; print_err "failed" ; return $state ;
}

# call with eval_state $(xyz)
function eval_state {
    local state=$?

    if (( $state == 0 ))
        then print_ok "success ${1}"
        else print_err "failed ${1}"
    fi

    return $state
}
