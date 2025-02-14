#!/bin/bash

#############################################################################################################################
# This component contains the logic that either terminates the pipeline, aborts experiments or waits for them.              #
# Depending on the setup, the further execution will be skipped or proceeded.                                               #
# Moreover, the component can wait for or abort experiments, if desired.                                                    #
# This functionality is needed to make the transition from per run to per experiment (as far the execution is async).       #
#############################################################################################################################

# invokation via wrapper pipe or directly calling the abort script
# call always the dump script afterwards if option is not exit

# daemon -> nohup/at/cronjob
# also manually trigger/entry

# use sigint vs. sigkill
# use countdown/count attempts?
# first try interrupt, after some time kill

# also make it possible to kill a local run via pid
# not only slurm ones with scancel

lookup="$_"
apply_config "${lookup^^}"


if [[ "${AWAIT_CONFIG['METHOD']}" == "exit" ]]
    then print_warn "script will be ended!"
    else print_warn "script will not be ended!"
fi

if [[ "${AWAIT_CONFIG['ABORT']}" != "never" ]]
    then print_warn "script will abort experiment!"
    else print_warn "script will not abort experiment!"
fi

if [[ "${AWAIT_CONFIG['HOLD']}" != "none" ]]
    then print_warn "script will terminate runs!"
    else print_warn "script will not terminate runs!"
fi

if [[ "${AWAIT_CONFIG['LIMIT']}" != "none" ]]
    then print_warn "script will limit waiting!"
    else print_warn "script will not limit waiting!"
fi


function await_progress {
    case "${COMMON_CONFIG['MODE']}" in
        "bash") ps -aux | grep "execute.bash" ;;
        "slurm") squeue --me --noheader --format="|%i|%j|%T|%N|" | grep -Pi "|.*|.*${EXP_NAME}.*|.*|.*|" ;;
    esac
}

function await_abort {
    case "${COMMON_CONFIG['MODE']}" in
        "bash") await_abort_bash "${1}" ;;
        "slurm") await_abort_slurm "${1}" ;;
    esac
}

function await_abort_bash {
    for entry in $(await_progress)
    do
        pid=$(echo $entry | awk '{print $1}')
        ppid=$(echo $entry | awk '{print $2}')
        usr=$(echo $entry | awk '{print $3}')
        cmd=$(echo $entry | awk '{print $4}')

        print_warn "abort pid=${pid}, ppid=${ppid}, usr=${usr}, cmd=${cmd}"
        if [[ "${AWAIT_CONFIG['HOLD']}" == "none" ]]
            then eval_state $(kill "${pid}")
            else eval_state $(kill -"${1}" "${pid}")
        fi
    done
}

function await_abort_slurm {
    for entry in $(await_progress)
    do
        id=$(echo $entry | awk -F'|' '{print $2}')
        name=$(echo $entry | awk -F'|' '{print $3}')
        state=$(echo $entry | awk -F'|' '{print $4}')
        node=$(echo $entry | awk -F'|' '{print $5}')

        print_warn "abort id=${id}, name=${name}, state=${state}, node=${node}"
        if [[ "${AWAIT_CONFIG['HOLD']}" == "none" ]]
            then eval_state $(scancel --user "${USER}" "${id}")
            else eval_state $(scancel --user "${USER}" --full --signal="${1}" "${id}")
        fi
    done
}

function await_try {
    if [[ "${AWAIT_CONFIG['LIMIT']}" == "none" ]]
        then await_abort "${signal}" ; return -1
    fi

    # couter and timer don't really make a difference in this way
    for _ in {$counter..0}
    do
        await_abort "${signal}"

        for _ in {$timer..0}
        do
            sleep 1

            if (( $(await_progress | wc -l) == 0 ))
                then return 0
            fi
        done
    done

    return 0
}

function await_trigger {
    case "${AWAIT_CONFIG['ABORT']}" in
        "never") return -1 ;;
        "change") if [[ "${1}" != "${2}" ]] ; then await_try ; return $? ; fi ;;
        "instant") await_try ; return $? ;;
    esac
}

function await_loop {
    until (( $(await_progress | wc -l) == 0 ))
    do
        prev=$(await_progress | wc -l)
        sleep 60 # check_interval=60
        next=$(await_progress | wc -l)

        # schauen, ob trigger da ist und dann versuchen abzubrechen
        await_trigger "${prev}" "${next}" && return
    done
}

#----------------------------------------------------------------------------------------------------------------------------

case "${AWAIT_CONFIG['METHOD']}" in
    "exit") exit ;; # could infere with the trap of the wrapper
    "pass") return ;; # ignore this component and triggers the next one
    "wait") : ;; # holds the whole logic of this component
esac

case "${AWAIT_CONFIG['LIMIT']}" in
    "none") : ;; # -> wait until all experiments are out of the queue/no longer running (e.g., externally killed)
    "single")   counter=1 ; timer=300 ;;
    "multiple") counter=3 ; timer=300 ;;
esac

case "${AWAIT_CONFIG['HOLD']}" in
    "none") : ;; # -> if scancel gets no signal, it will kill the process on a different level
    "gentle")   signal="SIGTERM" ;;
    "moderate") signal="SIGINT"  ;;
    "forceful") signal="SIGKILL" ;;
esac

await_loop

# using nohup (for the initial wrapper exec => exp-gen script)
# or using srun (for the initial wrapper exec => exp-gen script)
# in both cases it should be an backgroundjob

# start the whole script in background or let this component wait in background
# however, it is important, that this process is not somehow interrupted (beside a reboot etc.)
# the promt should be freed, so the user can continue to use the same shell

# test for each component, how the invokation was or ignor this, as they will be called as intended
# use only the COMMON_CONFIG['MODE'] to differ between cluster and manual use
# also container or not should be not relevant
