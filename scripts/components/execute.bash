#!/bin/bash

#############################################################################################################################
# This component contains the actual execution instructions and will execute several processes as background jobs.          #
# Depending on the setup, an additional debug node will be exeucted as well as gazebo in gui or headless mode.              #
# However, the default argparse setting should be defined here, this file can also be used for the experiment generation.   #
# This component also will provide the global EXIT_CODE, which says, if the execution was at least successfull.             #
#############################################################################################################################

# return a global state of each component instead the last return code
# enable optinal logging for each component, so its output is stored in a log file
# independent of the option, if the slurm takes the output
# make this setting comparable to the checks logic

# use nice to define the priority, but only possible as root
# also the ros2 rt implementation needs some source compiled parts and settings, which only root can make

lookup="$_"
apply_config "${lookup^^}"


if [[ "${EXECUTE_CONFIG['PROFILE']}" == "yes" ]]
    then print_warn "profiler will be started!"
    else print_warn "profiler will not be started!"
fi

if [[ "${EXECUTE_CONFIG['DEBUG']}" == "yes" ]]
    then print_warn "debug node will be started!"
    else print_warn "debug node will not be started!"
fi

if [[ "${EXECUTE_CONFIG['GAZEBO']}" == "gui" ]]
    then print_warn "gazebo gui will be started!"
    else print_warn "gazebo gui will not be started!"
fi

if [[ "${EXECUTE_CONFIG['PRIORITY']}" != "default" ]]
    then print_warn "execution priority will be default!"
    else print_warn "execution priority will not be default!"
fi


PROCESSES=(
    "ros2.*run"
    "ros_gz_bridge"
    "gz.*sim"
    "line_following.sdf"
    "gazebo_simulator"
    "profiler.py"
    "DebugInfos.py"
    "LFAgent.py"
)

WS_ENTRY="${SRC_PATH}/${CRL_REPOSITORY}/workspace"
CODE_ENTRY="${WS_ENTRY}/devel/src/${PROJECT}/${PROJECT}"


function execute_init {
    for workspace in $(ls "${WS_ENTRY}")
    do
        print_info "sourcing workspace ${workspace}"
        source "${WS_ENTRY}/${workspace}/install/setup.bash" ; eval_state
        # must be source in this shell, not in a subshell
    done
    print_warn "source ROS_OVERLAY -> ${COLCON_PREFIX_PATH}"
}

function execute_check {
    print_info "check process ${entry}"
    eval_state $(check_process "${entry}")
}

function execute_kill {
    print_info "try to kill ${entry}"
    eval_state $(kill_process "${entry}")
}

function execute_watchout {
    print_warn "watchout for possible zombies"
    for entry in ${PROCESSES[@]}
    do
        execute_check &&
        execute_kill
    done
}

function execute_state {
    state=$?
    if (( $state == 0 ))
        then print_ok "success (${1})"
        else print_err "failed (${1})"
    fi
    return $state
}

#----------------------------------------------------------------------------------------------------------------------------

execute_init

execute_watchout
print_info "timestamp before: $(create_timestamp)"

print_warn "setting trap for interruptions"
trap : SIGINT


if [[ "${EXECUTE_CONFIG['PROFILE']}" == "yes" ]]
    then
        print_info "starting profile script"
        {
            python3 -u "${SRC_PATH}/${CRL_REPOSITORY}/profiler.py" \
            --interval 10                                          \
            --chunk    1000                                        \
            --cpu      no                                          \
            --mem      no                                          \
            --gpu      no                                          \
            --blk      no                                          \
            --net      no                                          \
            --sens     no                                          \
            --pids     yes                                         \
            ; execute_state "profiler" ;
        } &
fi

# NOTE: Simulation starts on call!
# case "${EXECUTE_CONFIG['GAZEBO']}" in
#     "gui")      sim_options="-r" ;;
#     "headless") sim_options="-r -s --headless-rendering" ;;
# esac

case "${EXECUTE_CONFIG['GAZEBO']}" in
    "gui")      sim_options="" ;;
    "headless") sim_options="-s --headless-rendering" ;;
esac


print_info "starting Gazebo simulation"
{
    gz sim ${sim_options} "${SRC_PATH}/${CRL_REPOSITORY}/line_following.sdf" \
    ; execute_state "gazebo_simulator" ;
} &

# zum testen der topics
# print_info "starting ROS-GZ topic bridge"
# {
#     ros2 run ros_gz_bridge parameter_bridge \
#     /clock@ros_gz_interfaces/msg/Clock[ignition.msgs.Clock \
#     /vehicle/camera@sensor_msgs/msg/Image[ignition.msgs.Image \
#     /vehicle/motor@geometry_msgs/msg/Twist]ignition.msgs.Twist \
#     /vehicle/odo@nav_msgs/msg/Odometry[ignition.msgs.Odometry ;
#     --ros-args --params-file "${SRC_PATH}/${CRL_REPOSITORY}/qos.yml" \
#     ; execute_state "ros_gz_bridge" ;
# } &

# zum testen der services
# print_info "starting ROS-GZ service bridge"
# {
#     ros2 run ros_gz_bridge parameter_bridge \
#     /world/race_tracks_world/control@ros_gz_interfaces/srv/ControlWorld \
#     /world/race_tracks_world/set_pose@ros_gz_interfaces/srv/SetEntityPose \
#     ; execute_state "ros_gz_bridge" ;
# } &

print_info "starting ROS-GZ bridge"
{
    ros2 run ros_gz_bridge parameter_bridge \
    /clock@ros_gz_interfaces/msg/Clock[ignition.msgs.Clock \
    /vehicle/camera@sensor_msgs/msg/Image[ignition.msgs.Image \
    /vehicle/motor@geometry_msgs/msg/Twist]ignition.msgs.Twist \
    /vehicle/odo@nav_msgs/msg/Odometry[ignition.msgs.Odometry \
    /world/race_tracks_world/control@ros_gz_interfaces/srv/ControlWorld \
    /world/race_tracks_world/set_pose@ros_gz_interfaces/srv/SetEntityPose \
    /world/race_tracks_world/set_physics@ros_gz_interfaces/srv/AdjustPhysics \
    --ros-args --params-file "${SRC_PATH}/${CRL_REPOSITORY}/qos.yml" \
    ; execute_state "ros_gz_bridge" ;
} &

# print_info "starting ROS-GZ bridge"
# {
#     ros2 run ros_gz_bridge parameter_bridge --ros-args \
#     -p config_file:="${SRC_PATH}/${CRL_REPOSITORY}/config.yml" \
#     --params-file "${SRC_PATH}/${CRL_REPOSITORY}/qos.yml" \
#     ; execute_state "ros_gz_bridge" ;
# } &

if [[ "${EXECUTE_CONFIG['DEBUG']}" == "yes" ]]
    then
        print_info "starting debug script"d
        {
            python3 -u "${CODE_ENTRY}/debug/DebugInfos.py" \
            --odometry    1 6 6                            \
            --clock       0 0 0                            \
            --image       0 0 0                            \
            --twist       0 0 0                            \
            --observation 0 0 0                            \
            --action      0 0 0                            \
            --reward      0 0 0                            \
            ; execute_state "DebugInfos" ;
        } &
fi

print_info "starting gazebo simulation agent"
{
    python3 -u "${CODE_ENTRY}/agent/LFAgent.py"                                                                             \
        --fuck_it                                           no                                                              \
        --seed                                              42                                                              \
        --exp_id                                            "${EXP_ID}"                                                     \
        --root_dir                                          "${DAT_PATH}"                                                   \
        --debug                                             yes                                                             \
        --verbose                                           yes                                                             \
        --logging_mode                                      sync                                                            \
        --console_level                                     4                                                               \
        --file_level                                        4                                                               \
        --report_level                                      switch                                                          \
        --report_frequency                                  1                                                               \
        --report_entities                                                                                                   \
        --cpu_only                                          no                                                              \
        --train_subtasks                                    straight zero_3_r zero_3_l                                      \
        --eval_subtasks                                     straight zero_3_r zero_3_l                                      \
        --context_change                                    alternately                                                     \
        --context_reset                                     partial                                                         \
        --train_swap                                        1                                                               \
        --eval_swap                                         -1                                                              \
        --task_repetition                                   1                                                               \
        --begin_with                                        eval                                                            \
        --end_with                                          eval                                                            \
        --training_duration_unit                            timesteps                                                       \
        --evaluation_duration_unit                          timesteps                                                       \
        --training_duration                                 25000                                                           \
        --evaluation_duration                               250                                                             \
        --eval_random_track                                 no                                                              \
        --eval_random_position                              no                                                              \
        --eval_random_orientation                           no                                                              \
        --train_random_track                                no                                                              \
        --train_random_position                             no                                                              \
        --train_random_orientation                          no                                                              \
        --train_batch_size                                  32                                                              \
        --gamma                                             0.9                                                             \
        --algorithm                                         QGMM                                                            \
        --checkpointing                                     yes                                                             \
        --start_as_task                                     0                                                               \
        --load_ckpt                                         \
        --dqn_fc1_dims                                      256                                                             \
        --dqn_fc2_dims                                      128                                                             \
        --dqn_adam_lr                                       1e-3                                                            \
        --dqn_target_network                                yes                                                             \
        --dqn_target_network_update_freq                    1000                                                            \
        --dqn_dueling                                       yes                                                             \
        --qgmm_K                                            256                                                             \
        --qgmm_eps_0                                        0.011                                                           \
        --qgmm_eps_inf                                      0.01                                                            \
        --qgmm_lambda_sigma                                 0.                                                              \
        --qgmm_lambda_pi                                    0.                                                              \
        --qgmm_alpha                                        0.011                                                           \
        --qgmm_gamma                                        0.90                                                            \
        --qgmm_regEps                                       0.1                                                             \
        --qgmm_lambda_W                                     1.0                                                             \
        --qgmm_lambda_b                                     0.0                                                             \
        --qgmm_reset_somSigma                               0.4                                                             \
        --qgmm_somSigma_sampling                            yes                                                             \
        --qgmm_log_protos_each_n                            1000                                                            \
        --qgmm_init_forward                                 no                                                              \
        --exploration                                       eps-greedy                                                      \
        --initial_epsilon                                   1.0                                                             \
        --epsilon_delta                                     7e-5                                                            \
        --final_epsilon                                     0.01                                                            \
        --eps_replay_factor                                 1.0                                                             \
        --replay_buffer                                     default                                                         \
        --capacity                                          32                                                              \
        --per_alpha                                         0.6                                                             \
        --per_beta                                          0.6                                                             \
        --per_eps                                           1e-6                                                            \
        --processed_features                                gs                                                              \
        --sequence_stacking                                 v                                                               \
        --sequence_length                                   3                                                               \
        --input_shape                                       4 100                                                           \
        --output_shape                                      9                                                               \
        --line_mode                                         c                                                               \
        --line_detection                                    a                                                               \
        --line_threshold                                    0.25                                                            \
        --step_duration_nsec                                125e+6                                                          \
        --max_steps_per_episode                             10000                                                           \
        --max_steps_without_line                            1                                                               \
        --retry_attempts                                    no                                                              \
        --retry_boundary                                    5                                                               \
        --state_shape                                       4 100 3                                                         \
        --action_shape                                      2                                                               \
        --state_quantization                                -1.0 1.0 1000                                                   \
        --action_quantization                               0.2 0.5 3                                                       \
        --state_normalization                               -1.0 1.0                                                        \
        --action_normalization                              -1.0 1.0                                                        \
        --reward_terminal                                   -10.0                                                           \
        --reward_calculation                                s                                                               \
        --reward_calculation_weights                        1.0                                                             \
        --reward_clipping                                   no                                                              \
        --reward_clipping_range                             -1.0 1.0                                                        \
        --reward_normalization                              yes                                                             \
        --reward_normalization_range                        0.0 1.0                                                         \
    ; execute_state "LFAgent" ;
} &

print_info "set daemon for killswitch"
wait -n ; EXIT_CODE=$?
pkill -P $$ ; wait 10 ; pkill -9 -P $$

if (( $EXIT_CODE == 0 ))
    then print_ok "EXIT_CODE: ${EXIT_CODE}"
    else print_err "EXIT_CODE: ${EXIT_CODE}"
fi

print_warn "unsetting trap for interruptions"
trap - SIGINT

print_info "timestamp after: $(create_timestamp)"
execute_watchout

# execution not working, code raises an exception, but not traceback is thrown
# only a path issue or something different?!