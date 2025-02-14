import sys
import argparse


class LearnerArgs():
    def __new__(cls):
        try: return cls.__instance__
        except:
            cls.initialize_once()
            cls.__instance__ = super().__new__(cls)
            return cls.__instance__

    @classmethod
    def initialize_once(cls):
        tracks = ['straight', 'snake', 'puzzle', 'slalom_asym', 'slalom_sym', 'butterfly_rl', 'butterfly_lr', 'jojo_r', 'jojo_l', 'infinity_rl', 'infinity_lr']

        cls.parser = argparse.ArgumentParser('ICRL - Learner', 'Learner-based argumentparser of the ICRL-App.')

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
        execution_group.add_argument('--checkpointing', type=str, default='no', choices=['yes', 'no'], help='Enabled this mode to store RLLib / TF checkpoints after every subtask.')

        setup_group = cls.parser.add_argument_group('setup')
        setup_group.add_argument('--environment', type=str, default='LF',        choices=['LF'],        help='The environment (setup) which should be used to evaluate.')
        setup_group.add_argument('--scenario',  type=str, default='env-shift', choices=['env-shift'], help='The type of non-stationarity of the RL domain.')

        setting_group = cls.parser.add_argument_group('setting')
        setting_group.add_argument('--train_subtasks', nargs='*', type=str, default=['straight'],  choices=tracks,                        help='List of all subtasks that are used during the training.')
        setting_group.add_argument('--eval_subtasks',  nargs='*', type=str, default=['straight'],  choices=tracks,                        help='List of all subtasks that are used during the evaluation.')
        setting_group.add_argument('--context_change',            type=str, default='alternately', choices=['alternately', 'completely'], help='Defines how a context change is performed.')
        setting_group.add_argument('--context_reset',  nargs='?', type=str, default=None,          choices=['partial', 'full'],           help='Defines how a reset is performed with each context change.')
        setting_group.add_argument('--train_swap',                type=int, default=1,                                                    help='Sets the number of iterations before a context change occurs (train2eval).')
        setting_group.add_argument('--eval_swap',                 type=int, default=-1,                                                   help='Sets the number of iterations before a context change occurs (eval2train).')
        setting_group.add_argument('--task_repetition',           type=int, default=1,                                                    help='Each task is repeated N times.')
        setting_group.add_argument('--begin_with',                type=str, default='eval',        choices=['train', 'eval'],             help='Defines the mode the learner should start with.')
        setting_group.add_argument('--end_with',                  type=str, default='eval',        choices=['train', 'eval'],             help='Defines the mode the learner should finish with.')

        duration_group = cls.parser.add_argument_group('duration')
        duration_group.add_argument('--training_duration',        type=int, default=100000,                                         help='Defines the number of iterations á training_duration_unit.')
        duration_group.add_argument('--evaluation_duration',      type=int, default=10,                                             help='Defines the number of iterations á evaluation_duration_unit.')
        duration_group.add_argument('--training_duration_unit',   type=str, default='timesteps', choices=['timesteps', 'episodes'], help='Sets the unit (abstraction level) to determine what counts as an training iteration.')
        duration_group.add_argument('--evaluation_duration_unit', type=str, default='episodes',  choices=['timesteps', 'episodes'], help='Sets the unit (abstraction level) to determine what counts as an evaluation iteration.')

        randomization_group = cls.parser.add_argument_group('randomization')
        randomization_group.add_argument('--eval_random_track',        type=str, default='no', choices=['yes', 'no'], help='If inter task changes (context changes) should be randomized.')
        randomization_group.add_argument('--eval_random_spawn',        type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')
        randomization_group.add_argument('--eval_random_position',     type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')
        randomization_group.add_argument('--eval_random_orientation',  type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')
        randomization_group.add_argument('--train_random_track',       type=str, default='no', choices=['yes', 'no'], help='If inter task changes (context changes) should be randomized.')
        randomization_group.add_argument('--train_random_spawn',       type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')
        randomization_group.add_argument('--train_random_position',    type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')
        randomization_group.add_argument('--train_random_orientation', type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')

        learner_group = cls.parser.add_argument_group('learner')
        learner_group.add_argument('--optimizer',              type=str,   default='sgd',                                          help='The type of optimizer to calculate gradients.')
        learner_group.add_argument('--sgd_momentum',           type=float, default=0.,                                             help='The momentum used for sgd.')
        learner_group.add_argument('--adam_beta1',             type=float, default=0.9,                                            help='The first beta value used for adam.')
        learner_group.add_argument('--adam_beta2',             type=float, default=0.999,                                          help='The second beta value used for adam.')
        learner_group.add_argument('--lr',                     type=float, default=1e-03,                                          help='The (fixed) learning rate of the model.')
        learner_group.add_argument('--lr_schedule', nargs='*', type=str,   default=None,              help='Adaptive LR - pass a list containing tuples of (int, float) for (timestep, lr).')
        learner_group.add_argument('--train_batch_size',       type=int,   default=32,                                             help='Defines the mini-batch size that is used for inference.')
        learner_group.add_argument('--train_batch_iteration',  type=int,   default=1,                                              help='Defines how often the mini-batch is used for inference.')
        learner_group.add_argument('--update_freq',            type=int,   default=1,                                              help='Sets the number of steps after which the base model is updated.')
        learner_group.add_argument('--update_freq_unit',       type=str,   default='timesteps', choices=['timesteps', 'episodes'], help='Sets the unit (abstraction level) what counts as iteration for base model update.')
        learner_group.add_argument('--gamma',                  type=float, default=0.95,                                           help='The discount factor of the bellman equation.')

        agent_group = cls.parser.add_argument_group('agent')
        agent_group.add_argument('--algorithm',  type=str, default='QL',  choices=['QL', 'PG'],          help='Sets the RL algorithm to use.')
        agent_group.add_argument('--backend',    type=str, default='DQN', choices=['DQN', 'QGMM', 'PG'], help='Sets the learner backend for the algorithm.')
        agent_group.add_argument('--model_type', type=str, default='CNN', choices=['DNN', 'CNN', 'RNN'], help='Sets the model architecture for the learner backend.')

        policy_group = cls.parser.add_argument_group('policy')
        dqn_group = cls.parser.add_argument_group('dqn')
        dqn_group.add_argument('--double_q',                         type=str,   default='no',        choices=['yes', 'no'],              help='Use D-DQN?')
        dqn_group.add_argument('--dueling',                          type=str,   default='no',        choices=['yes', 'no'],              help='Use dueling DQNs?')
        dqn_group.add_argument('--training_intensity',    nargs='?', type=float, default=None,                                            help='The intensity with which to update the model (vs. collecting samples from the env).')
        dqn_group.add_argument('--target_network',                   type=str,   default='no',        choices=['yes', 'no'],              help='Defines if an additional target model should be employed.')
        dqn_group.add_argument('--target_network_update_freq',       type=int,   default=100,                                             help='Sets the number of steps after which the target model is updated.')
        dqn_group.add_argument('--target_network_update_freq_unit',  type=str,   default='timesteps', choices=['timesteps', 'episodes'],  help='Sets the unit (abstraction level) what counts as iteration for target model update.')
        dqn_group.add_argument('--tau',                              type=float, default=1.,                                              help='Defines some kind of Polyak averaging for the target model.')
        dqn_group.add_argument('--n_step',                           type=int,   default=1,                                               help='How many steps to look ahead before updating Q function.')
        dqn_group.add_argument('--noisy',                            type=str,   default='no',        choices=['yes', 'no'],              help='Whether to use noisy layers for DDQN.')
        dqn_group.add_argument('--noisy_sigma0',                     type=float, default=0.5,                                             help='Control the initial parameter noise for noisy nets.')
        dqn_group.add_argument('--td_error_loss_fn',                 type=str,   default='huber',                                         help='Defines the loss function that is used by the learner.')
        dqn_group.add_argument('--fcnet_hiddens',         nargs='+', type=int,   default=[64, 32],                                        help='Defines the number of hidden layers to be used for the advantage and value branch.')
        dqn_group.add_argument('--fcnet_activation',      nargs='?', type=str,   default=None,        choices=['linear', 'tanh', 'relu'], help='Activation function descriptor.')
        dqn_group.add_argument('--hiddens',               nargs='+', type=int,   default=[32],                                            help='List of layer-sizes after(!) the Advantages(A)/Value(V)-split. Hence, each of the A- and V- branches will have this structure of Dense layers. To define the NN before this A/V-split.')
        dqn_group.add_argument('--post_fcnet_hiddens',    nargs='*', type=int,   default=[],                                              help='Some default models support a final FC stack of n Dense layers with given activation.')
        dqn_group.add_argument('--post_fcnet_activation', nargs='?', type=str,   default=None,        choices=['linear', 'tanh', 'relu'], help='Activation function descriptor.')
        dqn_group.add_argument('--no_final_linear',                  type=str,   default='no',        choices=['yes', 'no'],              help='Whether to skip the final linear layer used to resize the hidden layer outputs to size `num_outputs`. If True, then the last hidden layer should already match num_outputs.')
        dqn_group.add_argument('--free_log_std',                     type=str,   default='no',        choices=['yes', 'no'],              help='For DiagGaussian action distributions, make the second half of the model outputs floating bias variables instead of state-dependent. This only has an effect is using the default fully connected net.')
        dqn_group.add_argument('--vf_share_layers',                  type=str,   default='yes',       choices=['yes', 'no'],              help='Whether layers should be shared for the value function.')

        qgmm_group = cls.parser.add_argument_group('qgmm')
        qgmm_group.add_argument('--qgmm_ltm_include_top',      type=str,   default='yes',        choices=['yes', 'no'],           help='Builds the LTM with or without a final readout layer.')
        qgmm_group.add_argument('--qgmm_somSigma_sampling',    type=str,   default='yes',        choices=['yes', 'no'],           help='Activate to uniformly sample from a radius around the BMU.')
        qgmm_group.add_argument('--qgmm_K',                    type=int,   default=100,                                           help='Number of K components to use for the GMM layer.')
        qgmm_group.add_argument('--qgmm_reset_factor',         type=float, default=0.1,                                           help='Resetting annealing radius which is set with each new sub-task.')
        qgmm_group.add_argument('--qgmm_alpha',                type=float, default=0.011,                                         help='Regularizer alpha.')
        qgmm_group.add_argument('--qgmm_gamma',                type=float, default=0.95,                                          help='Regularizer gamma.')
        qgmm_group.add_argument('--qgmm_lambda_b',             type=float, default=0.,                                            help='Coefficient for the bias-vector (0. = no bias vector).')
        qgmm_group.add_argument('--qgmm_regEps',               type=float, default=0.05,                                          help='Learning rate for the readout layer (SGD epsilon).')
        qgmm_group.add_argument('--qgmm_loss_fn',              type=str,   default='q_learning', choices=['q_learning', 'huber'], help='Loss function to be used for the readout layer.')
        qgmm_group.add_argument('--qgmm_load_ckpt', nargs='?', type=str,   default='',                                            help='Provide a path to a checkpoint file for a warm-start of the GMM.')

        exploration_group = cls.parser.add_argument_group('exploration')
        exploration_group.add_argument('--exploration', nargs='?', type=str, default=None, choices=['eps-greedy', 'param-noise', 'stochastic-sampling'], help='The exploration strategy the agent should use.')

        eps_greedy_group = cls.parser.add_argument_group('epsilon-greedy')
        eps_greedy_group.add_argument('--initial_epsilon',          type=float, default=1.0,                                                  help='The initial probability of choosing a random action.')
        eps_greedy_group.add_argument('--final_epsilon',            type=float, default=0.01,                                                 help='The lowest probability of choosing a random action.')
        eps_greedy_group.add_argument('--warmup_timesteps',         type=int,   default=1000,                                                 help='Steps with the initial exploration rate which do not decrease epsilon.')
        eps_greedy_group.add_argument('--epsilon_timesteps',        type=int,   default=80000,                                                help='Steps with epsilon as an exploration rate which decrease epsilon.')
        eps_greedy_group.add_argument('--interpolation', nargs='?', type=str,   default=None, choices=['poly', 'exp', 'poly_inv', 'exp_inv'], help='Sets the interpolation of a piecewise schedule for EpsilonGreedy exploration.')
        eps_greedy_group.add_argument('--schedule_power',           type=float, default=2.0,                                                  help='(Polynomial scheduling) Set the exponent of the interpolation function.')
        eps_greedy_group.add_argument('--schedule_decay',           type=float, default=0.01,                                                 help='(Exponential scheduling) Set the base of the interpolation function.')

        param_noise_group = cls.parser.add_argument_group('parameter-noise')
        param_noise_group.add_argument('--initial_stddev',   type=float, default=1.0,   help='The initial standard deviation of the parameter noise exploration.')
        param_noise_group.add_argument('--random_timesteps', type=int,   default=10000, help='Steps when to randomly choose an action.')

        replay_group = cls.parser.add_argument_group('experience replay')
        replay_group.add_argument('--replay_buffer',                 nargs='?', type=str,   default=None, choices=['reservoir', 'prioritized'], help='Replay buffer type to store experiences.')
        replay_group.add_argument('--num_steps_sampled_before_learning_starts', type=int,   default=0,                                          help='Number of timesteps to populate the buffer in advance before learning starts.')
        replay_group.add_argument('--replay_sequence_length',                   type=int,   default=1,                                          help='Number of subsequent samples replayed from the buffer.')
        replay_group.add_argument('--capacity',                                 type=int,   default=1000,                                       help='Buffer storage capacity.')
        replay_group.add_argument('--per_alpha',                                type=float, default=0.6,                                        help='Sets the degree of prioritization used by the buffer [0, 1].')
        replay_group.add_argument('--per_beta',                                 type=float, default=0.4,                                        help='Sets the degree of importance sampling to suppress the influence of gradient updates [0, 1].')
        replay_group.add_argument('--per_eps',                                  type=float, default=1e-06,                                      help='Epsilon to add to the TD errors when updating priorities.')

        non_inferred_group = cls.parser.add_argument_group('non-inferred')
        non_inferred_group.add_argument('--processed_features',      type=str, default='gs', choices=['bw', 'gs', 'rgb', 'rgba'], help='How the observed image shall be processed (black/white, gray-scale, red-green-blue, red-green-blue-alpha).')
        non_inferred_group.add_argument('--sequence_stacking',       type=str, default='h',  choices=['h', 'v'],                  help='How the observed image shall be processed (black/white, gray-scale, red-green-blue, red-green-blue-alpha).')
        non_inferred_group.add_argument('--sequence_length',         type=int, default=3,                                         help='Set the number of observations that count as a single sample/state.')
        non_inferred_group.add_argument('--input_shape',  nargs='+', type=int, default=[4, 100],                                  help='Set the number of actions the agent is able to perform with each step.')
        non_inferred_group.add_argument('--output_shape', nargs='+', type=int, default=[9],                                       help='Set the number of actions the agent is able to perform with each step.')

        cls.args = cls.parser.parse_args()
        # print('FRONTEND args:', len(vars(cls.args)))

        for attr in [
            'fuck_it', 'debug', 'verbose', 'cpu_only', 'checkpointing', 'eval_random_track', 'eval_random_spawn',
            'eval_random_position', 'eval_random_orientation', 'train_random_track', 'train_random_spawn',
            'train_random_position', 'train_random_orientation', 'double_q', 'dueling', 'target_network',
            'noisy', 'no_final_linear', 'free_log_std', 'vf_share_layers'
        ]:
            if hasattr(cls.args, attr): setattr(cls.args, attr, chk_bool(getattr(cls.args, attr), False))

        for attr in ['training_intensity']:
            if hasattr(cls.args, attr): setattr(cls.args, attr, chk_type(getattr(cls.args, attr), float, None))

        for attr in [
            'logging_mode', 'report_level', 'context_reset', 'fcnet_activation', 'post_fcnet_activation',
            'exploration', 'interpolation', 'replay_buffer'
        ]:
            if hasattr(cls.args, attr): setattr(cls.args, attr, chk_type(getattr(cls.args, attr), str, None))

        cls.args.lr_schedule = build((list, tuple), (int, float))(cls.args.lr_schedule) # HACK


class SimulationArgs():
    def __new__(cls, *args):
        try: return cls.__instance__
        except:
            cls.initialize_once()
            cls.__instance__ = super().__new__(cls)
            return cls.__instance__

    @classmethod
    def initialize_once(cls):
        cls.parser = argparse.ArgumentParser('ICRL - Simulation', 'Simulation-based argumentparser of the ICRL-App.')

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

        setup_group = cls.parser.add_argument_group('setup')
        setup_group.add_argument('--environment', type=str, default='LF',        choices=['LF'],        help='The environment (setup) which should be used to evaluate.')
        setup_group.add_argument('--scenario',  type=str, default='env-shift', choices=['env-shift'], help='The type of non-stationarity of the RL domain.')

        line_group = cls.parser.add_argument_group('line config')
        line_group.add_argument('--line_mode',      type=str,   default='c',  choices=['l', 'c', 'r'],      help='Defines how the robot should follow the line.')
        line_group.add_argument('--line_detection', type=str,   default='a',  choices=['a', 't', 'c', 'b'], help='Defines how the line image is processed.')
        line_group.add_argument('--line_threshold', type=float, default=0.25,                               help='Sets a threshold at which a pixel is being detected as belonging to the line [0, 1].')

        time_group = cls.parser.add_argument_group('time config')
        time_group.add_argument('--transition_timedelta', type=float, default=1/2,                              help='Defines the duration a step (action) should at least take (in seconds).')
        time_group.add_argument('--rtf_limits',  nargs=2, type=float, default=[.5, 10.],                        help='Forces the simulation to pause until an action is published.')
        time_group.add_argument('--adapt_rtf',            type=str,   default='no',      choices=['yes', 'no'], help='Forces the simulation to pause until an action is published.')
        time_group.add_argument('--default_rtf',          type=float, default=1.,                               help='Forces the simulation to pause until an action is published.')
        time_group.add_argument('--force_waiting',        type=str,   default='yes',     choices=['yes', 'no'], help='Forces the simulation to pause until an action is published.')

        horizon_group = cls.parser.add_argument_group('horizon config')
        horizon_group.add_argument('--max_steps_per_episode',  type=int, default=10000, help='Sets the number of steps after which an episode gets terminated.')
        horizon_group.add_argument('--max_steps_without_line', type=int, default=1,     help='Sets the number of steps the robot is allowed to drive without seeing the line before the episode is terminated.')

        setting_group = cls.parser.add_argument_group('setting')
        setting_group.add_argument('--sequence_length', type=int, default=3,                           help='Set the number of observations which count as a single sample/state.')
        setting_group.add_argument('--repeating_steps', type=int, default=1,                           help='Set the number of steps for which an action is repeated (without selecting a new one).')
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

        randomization_group = cls.parser.add_argument_group('randomization')
        randomization_group.add_argument('--random_inter_task',                 type=str, default='no', choices=['yes', 'no'],                                                 help='If inter task changes (context changes) should be randomized.')
        randomization_group.add_argument('--random_intra_task',                 type=str, default='no', choices=['yes', 'no'],                                                 help='If intra task changes (episode switches) should be randomized.')
        randomization_group.add_argument('--sensor_perturbations',   nargs='*', type=str, default=[],   choices=['faulty_pixels', 'stain', 'noisy', 'blurry', 'mirrored'], help='If intra task changes (episode switches) should be randomized.')
        randomization_group.add_argument('--actuator_perturbations', nargs='*', type=str, default=[],   choices=['facile', 'cumbersome', 'noisy', 'mirrored'],             help='If intra task changes (episode switches) should be randomized.')
        # manipulating the signals can emulate "real" malfunctions

        connection_group = cls.parser.add_argument_group('connection')
        connection_group.add_argument('--connection_latency', type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')
        connection_group.add_argument('--connection_jitter',  type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')
        connection_group.add_argument('--connection_loss',    type=str, default='no', choices=['yes', 'no'], help='If intra task changes (episode switches) should be randomized.')
        # skip frames or instructs

        non_inferred_group = cls.parser.add_argument_group('non-inferred')

        cls.args = cls.parser.parse_args()
        # print('BACKEND args:', len(vars(cls.args)))

        for attr in [
            'fuck_it', 'debug', 'verbose', 'adapt_rtf', 'force_waiting', 'retry_attempts', 'reward_clipping',
            'reward_normalization', 'random_inter_task', 'random_intra_task', 'latency', 'jitter', 'loss'
        ]:
            if hasattr(cls.args, attr): setattr(cls.args, attr, chk_bool(getattr(cls.args, attr), False))

        for attr in ['logging_mode', 'report_level', 'context_reset']:
            if hasattr(cls.args, attr): setattr(cls.args, attr, chk_type(getattr(cls.args, attr), str, None))


class EntryArgs():
    def __new__(cls):
        try: return cls.__instance__
        except:
            cls.initialize_once()
            cls.__instance__ = super().__new__(cls)
            return cls.__instance__

    @classmethod
    def initialize_once(cls):
        if sys.argv[0].find('RLLibAgent.py') != -1:
            cls.ENTRY = 'FRONTEND'
            cls.args = LearnerArgs().args # NOTE: for backward compatibility

            for name, value in vars(LearnerArgs().args).items():
                setattr(cls, name, value)

        if sys.argv[0].find('GazeboSim.py') != -1:
            cls.ENTRY = 'BACKEND'
            cls.args = SimulationArgs().args # NOTE: for backward compatibility

            for name, value in vars(SimulationArgs().args).items():
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

# def single_list(value):
#     return tuple_list(value, 1)
#
# def double_list(value):
#     return tuple_list(value, 2)
#
# def triple_list(value):
#     return tuple_list(value, 3)
#
# def quadruple_list(value):
#     return tuple_list(value, 4)


'''
sys.argv = ['RLLibAgent.py', '-h']
sys.argv = ['GazeboSim.py', '-h']

sys.argv = ['RLLibAgent.py']
sys.argv = ['GazeboSim.py']

EntryArgs()
print(EntryArgs().args)
print(vars(EntryArgs().args))
'''
