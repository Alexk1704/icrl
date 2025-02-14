### General layout: 2 packages, CODE: execute script & Python code for simulation and world, and ICRL: bash scripts and utilities, template for execute script and ROS bridge/ROS messages packages
### There is ONE central execute script in CODE that can be  parameter-varied. This script may source ICRL utility scripts to, e.g., construct a workspace on TMP
### In general, only the ICRL repo with be moved to /tmp because the CODE package is pure python, no compilation necessary
### CODE:-->src-->repo_name--><Python files>
### CODE:-->world.sdf
### CODE:-->models(optional)
### ICRL:-->scripts
###      -->workspace/src/-->ros_bridge
###                       -->messgae_types package
### In general, ICRL will never be touched except to extend the types that can be transported by the bridge!!

source scripts/utils/functions.bash
source scripts/utils/variables.bash


# ----------------- environ functions

echo funcdefs

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
    if [[ "yes" == "yes" ]]
        then export_readable "${1}" "${2}"
        else export_writable "${1}" "${2}"
    fi
}

function environ_export {
    if [[ ! -v "${1}" || ! $(declare -p "${1}") =~ "-x" ]]
        then
            environ_lock "${1}" "${2}"
            #print_info "create export: $(declare -p ${1})"
        else
                    echo "ignore export: $(declare -p ${1})"
                    environ_lock "${1}" "${2}"
                    echo "override export: $(declare -p ${1})"
    fi
}

#### --- environ

### --- excecute
echo execute

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

function execute_check {
    #print_info "check process ${entry}"
    eval_state $(check_process "${entry}")
}

function execute_kill {
    #print_info "try to kill ${entry}"
    eval_state $(kill_process "${entry}")
}

function execute_watchout {
    #print_warn "watchout for possible zombies"
    for entry in ${PROCESSES[@]}
    do
        execute_check &&
        execute_kill
    done
}

function execute_state {
    state=$?
    if (( $state == 0 ))
        then echo "success (${1})"
        else echo "failed (${1})"
    fi
    return $state
}


### ---execute

### --user config

echo userpath
unset PYTHONPATH
SRC_PATH="/home/gepperth/research/programming/python"
export GIT_PATH="/home/gepperth/research/programming/python"
CRL_REPOSITORY="icrl"
#sim_options=" -r -s --headless-rendering --render-engine ogre "
sim_options=" -r  "
CODE_ENTRY="${SRC_PATH}/${CRL_REPOSITORY}/workspace/devel/src/gazebo_sim/gazebo_sim"
EXP_ID=exp1
DAT_PATH="${SRC_PATH}"
PROJECT=icrl
export PYTHONPATH=$PYTHONPATH:${SRC_PATH}/${CRL_REPOSITORY}/workspace/devel/src/gazebo_sim
export PYTHONPATH=$PYTHONPATH:${SRC_PATH}/sccl/src
### --user

### ---necessary env variables

echo  env

# ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS
source /opt/ros/rolling/setup.bash
echo "source ROS_UNDERLAY -> ${AMENT_PREFIX_PATH}"

environ_export "ROS_DOMAIN_ID"                 "42"
environ_export "ROS_AUTOMATIC_DISCOVERY_RANGE" "LOCALHOST"
environ_export "RMW_IMPLEMENTATION"            "rmw_fastrtps_cpp"
# ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS - ROS

echo ROS

# GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ
environ_export "GZ_VERSION"           "8"
environ_export "GZ_DISTRO"            "harmonic"
environ_export "GZ_IP"                "127.0.0.1"
environ_export "GZ_PARTITION"         "$(hostname)"
environ_export "GZ_SIM_RESOURCE_PATH" "${GZ_SIM_RESOURCE_PATH:+${GZ_SIM_RESOURCE_PATH}:}${SRC_PATH}/${CRL_REPOSITORY}/models"
# GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ

echo WORKSPACE


source workspace/base/src/install/setup.bash
source workspace/devel/src/install/setup.bash


echo !!!!!!! $PYTHONPATH | tr ":" "\n"



execute_watchout

gz sim ${sim_options} "${SRC_PATH}/${CRL_REPOSITORY}/line_following.sdf"  &


    ros2 run ros_gz_bridge parameter_bridge \
    /clock@ros_gz_interfaces/msg/Clock[ignition.msgs.Clock \
    /vehicle/camera@sensor_msgs/msg/Image[ignition.msgs.Image \
    /vehicle/motor@geometry_msgs/msg/Twist]ignition.msgs.Twist \
    /vehicle/odo@nav_msgs/msg/Odometry[ignition.msgs.Odometry \
    /world/race_tracks_world/control@ros_gz_interfaces/srv/ControlWorld \
    /world/race_tracks_world/set_pose@ros_gz_interfaces/srv/SetEntityPose \
    /world/race_tracks_world/set_physics@ros_gz_interfaces/srv/AdjustPhysics \
    --ros-args --params-file "${SRC_PATH}/${CRL_REPOSITORY}/qos.yml"  &


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
        --checkpointing                                     no                                                              \
        --train_subtasks                                    jojo_l                                                          \
        --eval_subtasks                                     jojo_l                                                          \
        --context_change                                    alternately                                                     \
        --context_reset                                     partial                                                         \
        --train_swap                                        1                                                               \
        --eval_swap                                         -1                                                              \
        --task_repetition                                   1                                                               \
        --begin_with                                        train                                                           \
        --end_with                                          eval                                                            \
        --training_duration_unit                            timesteps                                                       \
        --evaluation_duration_unit                          timesteps                                                       \
        --training_duration                                 10000                                                           \
        --evaluation_duration                               1000                                                            \
        --eval_random_track                                 no                                                              \
        --eval_random_position                              no                                                              \
        --eval_random_orientation                           yes                                                             \
        --train_random_track                                no                                                              \
        --train_random_position                             no                                                              \
        --train_random_orientation                          yes                                                             \
        --train_batch_size                                  32                                                              \
        --gamma                                             0.95                                                            \
        --algorithm                                         QGMM                                                            \
        --dqn_fc1_dims                                      256                                                             \
        --dqn_fc2_dims                                      128                                                             \
        --dqn_adam_lr                                       1e-3                                                            \
        --dqn_target_network                                yes                                                             \
        --dqn_target_network_update_freq                    1000                                                            \
        --dqn_dueling                                       yes                                                             \
        --qgmm_K                                            81                                                              \
        --qgmm_eps_0                                        0.011                                                           \
        --qgmm_eps_inf                                      0.01                                                            \
        --qgmm_lambda_sigma                                 0.                                                              \
        --qgmm_lambda_pi                                    0.                                                              \
        --qgmm_alpha                                        0.011                                                           \
        --qgmm_gamma                                        0.9                                                             \
        --qgmm_regEps                                       0.01                                                            \
        --qgmm_lambda_W                                     1.0                                                             \
        --qgmm_lambda_b                                     0.0                                                             \
        --qgmm_reset_somSigma                               0.5 0.5                                                         \
        --qgmm_somSigma_sampling                            yes                                                             \
        --qgmm_log_protos_each_n                            1000                                                            \
        --qgmm_init_forward                                 no                                                              \
        --exploration                                       eps-greedy                                                      \
        --initial_epsilon                                   1.0                                                             \
        --epsilon_delta                                     0.000125                                                        \
        --final_epsilon                                     0.01                                                            \
        --replay_buffer                                     default                                                         \
        --capacity                                          5000                                                            \
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
        --max_steps_per_episode                             1000                                                            \
        --max_steps_without_line                            1                                                               \
        --retry_attempts                                    no                                                              \
        --retry_boundary                                    5                                                               \
        --state_shape                                       4 100 3                                                         \
        --action_shape                                      2                                                               \
        --state_quantization                                -1.0 1.0 1000                                                   \
        --action_quantization                               0.0 0.1 3                                                       \
        --state_normalization                               -1.0 1.0                                                        \
        --action_normalization                              -1.0 1.0                                                        \
        --reward_terminal                                   -1.0                                                            \
        --reward_calculation                                s                                                               \
        --reward_calculation_weights                        1.0                                                             \
        --reward_clipping                                   no                                                              \
        --reward_clipping_range                             -1.0 1.0                                                        \
        --reward_normalization                              yes                                                             \
        --reward_normalization_range                        0.0 1.0                                                         




#        --qgmm_load_ckpt                                    /home/ak/Desktop/ckpts/icrl/gs/k81/lf-straight-4096             \


execute_watchout
