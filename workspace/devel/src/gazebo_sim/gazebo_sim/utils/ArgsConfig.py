import argparse


class Args():
    def __new__(cls):
        try: return cls.__instance__
        except:
            cls.initialize_once()
            cls.__instance__ = super().__new__(cls)
            return cls.__instance__

    @classmethod
    def initialize_once(cls):
        # tracks = ['straight', 'snake', 'puzzle', 'slalom_asym', 'slalom_sym', 'butterfly_rl', 'butterfly_lr', 'jojo_r', 'jojo_l', 'infinity_rl', 'infinity_lr']
        tracks = ['straight', 'zero_1_l', 'zero_2_l', 'zero_3_l', 'zero_4_l', 'zero_5_l', 'zero_6_l', 'circle_1_l', 'circle_2_l', 'circle_3_l', 'circle_4_l', 'circle_5_l', 'circle_6_l',
                    'zero_1_r', 'zero_2_r', 'zero_3_r', 'zero_4_r', 'zero_5_r', 'zero_6_r', 'circle_1_r', 'circle_2_r', 'circle_3_r', 'circle_4_r', 'circle_5_r', 'circle_6_r']
        cls.parser = argparse.ArgumentParser('ICRL', 'argparser of the ICRL-App.')

        # ------------------------------------ LEARNER

        default_group = cls.parser.add_argument_group('default')
        default_group.add_argument('--fuck_it',  type=str, default='no', choices=['yes', 'no'], help='Minimal pipeline settings for a quick run-through (use this for testing).')
        default_group.add_argument('--seed',     type=int, default=42,                          help='The random seed for the experiment run.')
        default_group.add_argument('--exp_id',   type=str, default='exp_id',                    help='Name of the experiment to use as an identifier for the generated results.')
        default_group.add_argument('--root_dir', type=str, default='./',                        help='Directory where all experiment results and logs are stored.')

        output_group = cls.parser.add_argument_group('output')
        output_group.add_argument('--debug',                   type=str, default='no', choices=['yes', 'no'],     help='Enable this mode to receive even more extensive output.')
        output_group.add_argument('--verbose',                 type=str, default='no', choices=['yes', 'no'],     help='Enable this mode to receive even more extensive output.')
        output_group.add_argument('--logging_mode', nargs='?', type=str, default=None, choices=['sync', 'async'], help='Set the mode of the main logger to control sync/async execution.')
        output_group.add_argument('--console_level',           type=int, default=0,    choices=range(5),          help='Set the level of verbosity to control the amount of console output.')
        output_group.add_argument('--file_level',              type=int, default=0,    choices=range(5),          help='Set the level of verbosity to control the amount of file output.')

        reporting_group = cls.parser.add_argument_group('reporting')
        reporting_group.add_argument('--report_level',    nargs='?', type=str, default=None, choices=['step', 'reset', 'switch', 'overall'], help='Frequency how often experiment results are stored.')
        reporting_group.add_argument('--report_frequency',           type=int, default=1,                                                    help='Frequency how often experiment results are stored.')
        reporting_group.add_argument('--report_entities', nargs='*', type=str, default=[],                                                   help='Entities of the experiment, which should be stored as results.')

        execution_group = cls.parser.add_argument_group('execution')
        execution_group.add_argument('--cpu_only',      type=str, default='no', choices=['yes', 'no'], help='Decides if the RLLib and TF code should be executed only via CPU rather than using the GPU.')
        
        lf_setting_group = cls.parser.add_argument_group('line following')
        lf_setting_group.add_argument('--train_subtasks', nargs='*', type=str, default=['straight'],  choices=tracks,   help='List of all subtasks that are used during the training.')
        lf_setting_group.add_argument('--eval_subtasks',  nargs='*', type=str, default=['straight'],  choices=tracks,   help='List of all subtasks that are used during the evaluation.')

        setting_group = cls.parser.add_argument_group('setting')
        setting_group.add_argument('--context_change',            type=str, default='completely',  choices=['alternately', 'completely'],   help='Defines how a context change is performed.')
        setting_group.add_argument('--context_reset',  nargs='?', type=str, default=None,          choices=['partial', 'full'],             help='Defines how a reset is performed with each context change.')
        setting_group.add_argument('--train_swap',                type=int, default=1,                                                      help='Sets the number of iterations before a context change occurs (train2eval).')
        setting_group.add_argument('--eval_swap',                 type=int, default=-1,                                                     help='Sets the number of iterations before a context change occurs (eval2train).')
        setting_group.add_argument('--task_repetition',           type=int, default=1,                                                      help='Each task is repeated N times.')
        setting_group.add_argument('--begin_with',                type=str, default='train',       choices=['train', 'eval'],               help='Defines the mode the learner should start with.')
        setting_group.add_argument('--end_with',                  type=str, default='eval',        choices=['train', 'eval'],               help='Defines the mode the learner should finish with.')

        duration_group = cls.parser.add_argument_group('duration')
        duration_group.add_argument('--training_duration',        type=int, default=10000, help='Defines the number of iterations á training_duration_unit.')
        duration_group.add_argument('--evaluation_duration',      type=int, default=10,    help='Defines the number of iterations á evaluation_duration_unit.')
        duration_group.add_argument('--training_duration_unit',   type=str, default='timesteps', choices=['timesteps', 'episodes'], help='Sets the unit (abstraction level) to determine what counts as an training iteration.')
        duration_group.add_argument('--evaluation_duration_unit', type=str, default='episodes',  choices=['timesteps', 'episodes'], help='Sets the unit (abstraction level) to determine what counts as an evaluation iteration.')

        randomization_group = cls.parser.add_argument_group('randomization')
        randomization_group.add_argument('--eval_random_track',        type=str, default='no', choices=['yes', 'no'], help='If inter task changes (context changes) should be randomized.')
        randomization_group.add_argument('--eval_random_position',     type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')
        randomization_group.add_argument('--eval_random_orientation',  type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')
        randomization_group.add_argument('--train_random_track',       type=str, default='no', choices=['yes', 'no'], help='If inter task changes (context changes) should be randomized.')
        randomization_group.add_argument('--train_random_position',    type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')
        randomization_group.add_argument('--train_random_orientation', type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')

        learner_group = cls.parser.add_argument_group('learner')
        learner_group.add_argument('--train_batch_size',       type=int,   default=32,      help='Defines the mini-batch size that is used for training.')
        learner_group.add_argument('--train_batch_iteration',  type=int,   default=1,       help='Defines how often the mini-batch is used for training.')
        learner_group.add_argument('--gamma',                  type=float, default=0.95,    help='The discount factor of the bellman equation.')

        agent_group = cls.parser.add_argument_group('agent')
        agent_group.add_argument('--algorithm',  type=str, default='DQN',  choices=['DQN', 'QGMM'], help='Sets the RL algorithm to use.')
        agent_group.add_argument('--model_type', type=str, default='DNN', choices=['DNN', 'CNN'],   help='Sets the model architecture for the learner backend.')
        agent_group.add_argument('--checkpointing', type=str, default='no', choices=['yes', 'no'],  help='Enable this to store TF model checkpoints after every training task.')
        agent_group.add_argument('--load_ckpt',  nargs='?', type=str, default='',                   help='Provide a path to a checkpoint file to initialize a model.')
        agent_group.add_argument('--start_as_task', type=int, default=0,                            help='Task identifier to start with.')

        dqn_group = cls.parser.add_argument_group('dqn')
        dqn_group.add_argument('--dqn_fc1_dims',                    type=int,   default=128,                                    help='Size of FC layer 1.')
        dqn_group.add_argument('--dqn_fc2_dims',                    type=int,   default=64,                                     help='Size of FC layer 2.')
        dqn_group.add_argument('--dqn_adam_lr',                     type=float, default=1e-3,                                   help='Learning rate for ADAM opt.')
        dqn_group.add_argument('--dqn_dueling',                     type=str,   default='no',        choices=['yes', 'no'],     help='Use dueling DQNs?')
        dqn_group.add_argument('--dqn_target_network',              type=str,   default='no',        choices=['yes', 'no'],     help='Whether to use double DQN (target network).')
        dqn_group.add_argument('--dqn_target_network_update_freq',  type=int,   default=1000,                                   help='Sets the number of steps after which the target model is updated.')

        qgmm_group = cls.parser.add_argument_group('qgmm')
        qgmm_group.add_argument('--qgmm_K',                         type=int,   default=81,                                     help='Number of K components to use for the GMM layer.')
        qgmm_group.add_argument('--qgmm_eps_0',                     type=float, default=0.011,                                  help='Start epsilon value (initial learning rate).')                                    
        qgmm_group.add_argument('--qgmm_eps_inf',                   type=float, default=0.01,                                   help='Smallest epsilon value (learning rate) for regularization.')                                    
        qgmm_group.add_argument('--qgmm_lambda_sigma',              type=float, default=0.,                                     help='Sigma factor.')                  
        qgmm_group.add_argument('--qgmm_lambda_pi',                 type=float, default=0.,                                     help='Pis factor.')                  
        qgmm_group.add_argument('--qgmm_alpha',                     type=float, default=0.01,                                   help='Regularizer alpha.')
        qgmm_group.add_argument('--qgmm_gamma',                     type=float, default=0.9,                                    help='Regularizer gamma.')
        qgmm_group.add_argument('--qgmm_regEps',                    type=float, default=0.01,                                   help='Learning rate for the readout layer (SGD epsilon).')
        qgmm_group.add_argument('--qgmm_lambda_W',                  type=float, default=1.,                                     help='Weight factor.')
        qgmm_group.add_argument('--qgmm_lambda_b',                  type=float, default=0.,                                     help='Bias factor.')
        qgmm_group.add_argument('--qgmm_reset_somSigma',            nargs='+',  type=float, default=[0.1],                      help='Resetting annealing radius which is set with each new sub-task.')      
        qgmm_group.add_argument('--qgmm_somSigma_sampling',         type=str,   default='yes',        choices=['yes', 'no'],    help='Activate to uniformly sample from a radius around the BMU.')
        qgmm_group.add_argument('--qgmm_log_protos_each_n',         type=int,   default=0,                                      help='Saves protos (as an image) each N steps.')

        qgmm_group.add_argument('--qgmm_init_forward',              type=str,   default='yes',        choices=['yes', 'no'],    help='Init weights to favor driving forward.')

        exploration_group = cls.parser.add_argument_group('exploration')
        exploration_group.add_argument('--exploration', nargs='?',  type=str, default=None, choices=['eps-greedy'], help='The exploration strategy the agent should use.')

        eps_greedy_group = cls.parser.add_argument_group('epsilon-greedy')
        eps_greedy_group.add_argument('--initial_epsilon',          type=float, default=1.0,    help='The initial probability of choosing a random action.')
        eps_greedy_group.add_argument('--final_epsilon',            type=float, default=0.01,   help='The lowest probability of choosing a random action.')
        eps_greedy_group.add_argument('--epsilon_delta',            type=float, default=0.001,  help='Epsilon reduction factor (stepwise).')
        eps_greedy_group.add_argument('--eps_replay_factor',        type=float, default=1.0,    help='Scaling factor for replay training.')

        replay_group = cls.parser.add_argument_group('experience replay')
        replay_group.add_argument('--replay_buffer',                 nargs='?', type=str,   default='default', choices=['default', 'prioritized'],  help='Replay buffer type to store experiences.')
        replay_group.add_argument('--capacity',                                 type=int,   default=1000,                                           help='Buffer storage capacity.')
        replay_group.add_argument('--per_alpha',                                type=float, default=0.6,                                            help='Sets the degree of prioritization used by the buffer [0, 1].')
        replay_group.add_argument('--per_beta',                                 type=float, default=0.4,                                            help='Sets the degree of importance sampling to suppress the influence of gradient updates [0, 1].')
        replay_group.add_argument('--per_eps',                                  type=float, default=1e-06,                                          help='Epsilon to add to the TD errors when updating priorities.')

        non_inferred_group = cls.parser.add_argument_group('non-inferred')
        non_inferred_group.add_argument('--processed_features',      type=str, default='gs', choices=['bw', 'gs', 'rgb', 'rgba'], help='How the observed image shall be processed (black/white, gray-scale, red-green-blue, red-green-blue-alpha).')
        non_inferred_group.add_argument('--sequence_stacking',       type=str, default='v',  choices=['h', 'v'],                  help='How the observed image shall be processed (black/white, gray-scale, red-green-blue, red-green-blue-alpha).')
        non_inferred_group.add_argument('--sequence_length',         type=int, default=3,                                         help='Set the number of observations that count as a single sample/state.')
        non_inferred_group.add_argument('--input_shape',  nargs='+', type=int, default=[4, 100],                                  help='Set the number of actions the agent is able to perform with each step.')
        non_inferred_group.add_argument('--output_shape', nargs='+', type=int, default=[9],                                       help='Set the number of actions the agent is able to perform with each step.')
        non_inferred_group.add_argument('--action_start_value',      required=False, type=float,     default=0.1,                 help='Starting val for action space generation')
        non_inferred_group.add_argument('--action_stop_value',       required=False, type=float,     default=0.6,                 help='Starting val for action space generation')
        non_inferred_group.add_argument('--action_samples',          required=False, type=int,       default=2,                   help='How many action values per wheel')
        non_inferred_group.add_argument('--obs_per_sec_sim_time',    required=False, type=int,       default=20,                  help='What is the SDF setting for camera frequency IN SIM TIME?')
        non_inferred_group.add_argument('--select_action_entries',   required=False, type=list[int], default=None,                help='From combinatorial actions: which opnes to actually use?')
        non_inferred_group.add_argument('--load_task',               required=False, type=int,       default=0,                   help='At what task to start and load the ml model')

        # ------------------------------------ SIMULATION

        line_group = cls.parser.add_argument_group('line config')
        line_group.add_argument('--line_mode',      type=str,   default='c',  choices=['l', 'c', 'r'],      help='Defines how the robot should follow the line.')
        line_group.add_argument('--line_detection', type=str,   default='a',  choices=['a', 't', 'c', 'b'], help='Defines how the line image is processed.')
        line_group.add_argument('--line_threshold', type=float, default=0.25,                               help='Sets a threshold at which a pixel is being detected as belonging to the line [0, 1].')

        time_group = cls.parser.add_argument_group('time config')
        time_group.add_argument('--transition_timedelta',   type=float, default=1/2,                                help='Defines the duration a step (action) should at least take (in seconds).')
        time_group.add_argument('--step_duration_nsec',     type=float, default=5e+8,                               help='frequency in simulation time is 1/step_duration_nsec')
        time_group.add_argument('--adapt_rtf',              type=str,   default='no',      choices=['yes', 'no'],   help='Activate dynamic real-time factor.')
        time_group.add_argument('--rtf_limits',  nargs=2,   type=float, default=[.5, 10.],                          help='Lower & higher bounds for real-time factor.')
        time_group.add_argument('--default_rtf',            type=float, default=.5,                                 help='Default real-time factor.')

        horizon_group = cls.parser.add_argument_group('horizon config')
        horizon_group.add_argument('--max_steps_per_episode',  type=int, default=10000, help='Sets the number of steps after which an episode gets terminated.')
        horizon_group.add_argument('--max_steps_without_line', type=int, default=1,     help='Sets the number of steps the robot is allowed to drive without seeing the line before the episode is terminated.')

        setting_group.add_argument('--retry_attempts',  type=str, default='no', choices=['yes', 'no'], help='Defines if the agent is allowed to retry its attempt and defines the type of reset to perform.')
        setting_group.add_argument('--retry_boundary',  type=int, default=5,                           help='Sets the threshold of retries for each reset method.')

        space_group = cls.parser.add_argument_group('state-action space')
        space_group.add_argument('--state_shape',          nargs='+', type=int,   default=[4, 100, 3],    help='The unflatted (nested) shape of the state message data.')
        space_group.add_argument('--action_shape',         nargs='+', type=int,   default=[2],            help='The unflatted (nested) shape of the action message data.')
        space_group.add_argument('--state_quantization',   nargs=3,   type=float, default=[0., 1., 1000], help='Creates the quantized state space by [lower, upper, step].')
        space_group.add_argument('--action_quantization',  nargs=3,   type=float, default=[0., 0.5, 3],   help='Creates the quantized action space by [lower, upper, step].')
        space_group.add_argument('--state_normalization',  nargs=2,   type=float, default=[-1., +1.],     help='How the state should be normalized (mainly for evaluation purposes).')
        space_group.add_argument('--action_normalization', nargs=2,   type=float, default=[-1., +1.],     help='How the action should be normalized (mainly for evaluation purposes).')

        reward_group = cls.parser.add_argument_group('reward composition')
        reward_group.add_argument('--reward_terminal',                       type=float, default=-10.,                                                   help='The reward the agent gets for failing irreversibly.')
        reward_group.add_argument('--reward_calculation',         nargs='+', type=str,   default=['s', 'a'], choices=['s', 'a', 'sd', 'ad', 'si', 'ai'], help='Defines how the reward function is composed (raw, deviation, improvement).')
        reward_group.add_argument('--reward_calculation_weights', nargs='+', type=float, default=[0.5, 0.5],                                             help='Defines how the reward function components are weighted.')
        reward_group.add_argument('--reward_clipping',                       type=str,   default='no',       choices=['yes', 'no'],                      help='If the rewards should be clipped or not.')
        reward_group.add_argument('--reward_clipping_range',      nargs=2,   type=float, default=[-1., +1.],                                             help='The lower and upper bound to be clipped.')
        reward_group.add_argument('--reward_normalization',                  type=str,   default='no',       choices=['yes', 'no'],                      help='If the rewards should be normalized or not.')
        reward_group.add_argument('--reward_normalization_range', nargs=2,   type=float, default=[-1., +1.],                                             help='The lower and upper bound to be normalized.')

        cls.args = cls.parser.parse_args()
        
        for attr in [
            'fuck_it', 'debug', 'verbose', 'cpu_only', 'checkpointing', 'eval_random_track', 'eval_random_spawn',
            'eval_random_position', 'eval_random_orientation', 'train_random_track', 'train_random_spawn',
            'train_random_position', 'train_random_orientation', 'dqn_target_network', 'dqn_dueling',
            'adapt_rtf', 'retry_attempts', 'reward_clipping', 'reward_normalization']:
            if hasattr(cls.args, attr): setattr(cls.args, attr, chk_bool(getattr(cls.args, attr), False))

        for attr in [
            'logging_mode', 'report_level', 'context_reset', 'exploration', 'interpolation', 'replay_buffer'
        ]:
            if hasattr(cls.args, attr): setattr(cls.args, attr, chk_type(getattr(cls.args, attr), str, None))

        cls.ENTRY = 'ALL'
        
        for name, value in vars(cls.args).items():
            setattr(cls, name, value)


# LOOKUPS = {
#     None: None,
#     bool: bool(),
#     int: int(),
#     float: float(),
#     str: str(),
#     tuple: tuple(),
#     set: set(),
#     list: list(),
#     dict: dict(),
# }

def to_bool(value, fallback):
    if value == 'yes': return True
    elif value == 'no': return False
    else: return fallback

def chk_bool(value, fallback):
    if isinstance(value, bool): return value
    else: return to_bool(value, fallback)

def to_type(value, dtype, fallback):
    try: return dtype(value)
    except: return fallback

def chk_type(value, dtype, fallback):
    if isinstance(value, dtype): return value
    else: return to_type(value, dtype, fallback)

def build(iterables, dtypes):
    def closure(values):
        if isinstance(values, str): values = values.split(' ')

        if len(values) == 0: return None
        if len(values) % len(dtypes) != 0: return None

        return zip(*[iter(values)] * len(dtypes))

    def flat_closure(values):
        wrapper, = iterables
        entries = closure(values)

        if entries is None: return None
        try: return wrapper(d(v) for entry in entries for d, v in zip(dtypes, entry))
        except: return None

    def nested_closure(values):
        outer, inner = iterables
        entries = closure(values)

        if entries is None: return None
        try: return outer(inner(d(v) for d, v in zip(dtypes, entry)) for entry in entries)
        except: return None

    if len(iterables) == 1: return flat_closure
    if len(iterables) == 2: return nested_closure
