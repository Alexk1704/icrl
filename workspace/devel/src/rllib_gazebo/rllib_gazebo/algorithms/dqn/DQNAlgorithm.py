import os
import numpy as np
import tree


from collections import defaultdict
from pprint import pprint
from typing import List, Optional, Type, Union, Callable

from ray.tune.logger import Logger


from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.simple_q.simple_q import SimpleQConfig

from ray.rllib.evaluation.metrics import (
    collect_episodes,
    collect_metrics
)
from ray.rllib.evaluation.worker_set import WorkerSet

from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step, multi_gpu_train_one_step

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch

from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration import StochasticSampling, ParameterNoise
from ray.rllib.utils.typing import ResultDict

from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED_THIS_ITER,
    NUM_ENV_STEPS_SAMPLED_THIS_ITER,
    LAST_TARGET_UPDATE_TS,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_TARGET_UPDATES,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
)

from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.replay_buffers.utils import (
    sample_min_n_steps_from_buffer,
    update_priorities_in_replay_buffer,
    validate_buffer_config,
)

from ray.rllib.utils.schedules import PiecewiseSchedule

from rllib_gazebo.algorithms.dqn.DQNPolicy import DQNPolicy
from rllib_gazebo.algorithms.utils.exploration.EpsGreedy import EpsilonGreedy

from rllib_gazebo.utils.Evaluation import *
from rllib_gazebo.utils.ArgsConfig import LearnerArgs
from rllib_gazebo.utils.Exchanging import Exchanger
from rllib_gazebo.utils.Logging import Logger
from rllib_gazebo.utils.Caching import Cache


class DQNAlgoConfig(AlgorithmConfig):
    ''' This DQN algorithm config is a mix of the SimpleQConfig & DQNConfig. '''
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or DQN)

        largs = LearnerArgs().args

        self.fuck_it = largs.fuck_it
        self.debug = largs.debug
        self.rollout_fragment_length = "auto" # Set to `self.n_step`, if 'auto'.

        exp_id = largs.exp_id
        root_dir = largs.root_dir
        self.results_dir = os.path.join(root_dir, exp_id, 'debug')

        self.training_duration = largs.training_duration
        self.training_duration_unit = largs.training_duration_unit

        # --------------> `training()`
        self.grad_clip = 40.0
        self.grad_clip_by = "global_norm"

        # --------------> `exploration()`
        self.explore = False
        self.exploration_config = {}
        if largs.exploration == 'eps-greedy':
            schedule = None
            interpolations = {
                'poly': lambda left, right, alpha: np.array(np.add(right, np.multiply(np.subtract(left, right), np.power(1 - alpha, largs.schedule_power))), dtype=np.float32),
                'exp': lambda left, right, alpha: np.array(np.add(right, np.multiply(np.subtract(left, right), np.power(largs.schedule_decay, alpha))), dtype=np.float32),
                'poly_inv': lambda left, right, alpha: np.array(np.add(left, np.multiply(np.subtract(right, left), np.power(alpha, largs.schedule_power))), dtype=np.float32),
                'exp_inv': lambda left, right, alpha: np.array(np.add(left, np.multiply(np.subtract(right, left), np.power(largs.schedule_decay, (1 - alpha)))), dtype=np.float32),
            }
            if largs.interpolation in interpolations:
                schedule = PiecewiseSchedule(framework='tf2', interpolation=interpolations[largs.interpolation], outside_value=largs.final_epsilon,
                    endpoints=[
                        (0, largs.initial_epsilon),
                        (largs.warmup_timesteps, largs.initial_epsilon),
                        (largs.warmup_timesteps + largs.epsilon_timesteps, largs.final_epsilon),
                    ],
                )

            self.explore = True
            self.exploration_config.update({
                'type': EpsilonGreedy,
                'initial_epsilon': largs.initial_epsilon,
                'final_epsilon': largs.final_epsilon,
                'warmup_timesteps': largs.warmup_timesteps,
                'epsilon_timesteps': largs.epsilon_timesteps,
                'epsilon_schedule': schedule,
            })
        elif largs.exploration == 'param-noise':
            self.explore = True
            self.exploration_config.update({
                'type': ParameterNoise,
                'initial_stddev': largs.initial_stddev,
                'random_timesteps': largs.random_timesteps,
            })

        # --------------> DQN
        self.training_intensity = largs.training_intensity                              # Defaults to None

        self.target_network = largs.target_network                                      # Whether to use an additional target network
        self.target_network_update_freq = largs.target_network_update_freq              # Sets the number of steps after which the target model is updated.
        self.target_network_update_freq_unit = largs.target_network_update_freq_unit    # Sets the unit (abstraction level) what counts as iteration for target model update.

        self.hiddens = largs.hiddens
        self.n_step = largs.n_step                          # N-step for Q-learning.
        self.td_error_loss_fn = largs.td_error_loss_fn      # 'huber' or 'mse'

        self.double_q = largs.double_q
        self.dueling = largs.dueling

        self.noisy = largs.noisy                            # Whether to use noisy network to aid exploration. This adds parametric noise to the model weights.
        self.sigma0 = largs.noisy_sigma0                    # Control the initial parameter noise for noisy nets.

        # ---> SimpleQ defaults
        self.num_steps_sampled_before_learning_starts = largs.num_steps_sampled_before_learning_starts
        self.train_batch_size = largs.train_batch_size

        self.lr = largs.lr
        self.lr_schedule = largs.lr_schedule

        self.adam_beta1 = largs.adam_beta1
        self.adam_beta2 = largs.adam_beta2

        self.tau = largs.tau                                # Defines some kind of Polyak averaging for the target model.

        # `reporting()`
        self.min_time_s_per_iteration = None
        self.min_train_timesteps_per_iteration = None
        self.min_sample_timesteps_per_iteration = None
        self.store_buffer_in_checkpoints = False

        # ---> DQN defaults (not used by argparser)
        self.before_learn_on_batch = None                   # Callback to run before learning on a multi-agent batch of experiences.
        self.num_atoms = 1                                  # Number of atoms for representing the distribution of return. When this is greater than 1, distributional Q-learning is used. (UNUSED)
        self.categorical_distribution_temperature = 1.0     # Used for categorical action dist. (UNUSED)
        self.v_min = -10.                                   # Minimum value estimation
        self.v_max = +10.                                   # Maximum value estimation

        # --------------> ReplayBuffer
        self.replay_buffer_choice = largs.replay_buffer
        self.replay_buffer_config = {
            # The number of continuous environment steps to replay at once.
            # This may be set to greater than 1 to support recurrent models.
            'replay_sequence_length': largs.replay_sequence_length,
            'capacity': largs.capacity,
            # Whether to compute priorities on workers.
            'worker_side_prioritization': False,
            'storage_unit': 'timesteps',  # 'timesteps', 'sequences', 'episodes'
        }
        if largs.replay_buffer == 'prioritized':
            # Replay buffer configuration.
            self.replay_buffer_config = {
                "type": "MultiAgentPrioritizedReplayBuffer",
                "prioritized_replay": DEPRECATED_VALUE,
                "prioritized_replay_alpha": largs.per_alpha,
                # Beta parameter for sampling from prioritized replay buffer.
                "prioritized_replay_beta": largs.per_beta,
                # Epsilon to add to the TD errors when updating priorities.
                "prioritized_replay_eps": largs.per_eps,
            }
        else:
            self.replay_buffer_config.update({
                'type': 'ReservoirReplayBuffer',
            })

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        target_network_update_freq: Optional[int] = NotProvided,
        replay_buffer_config: Optional[dict] = NotProvided,
        store_buffer_in_checkpoints: Optional[bool] = NotProvided,
        lr_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        adam_beta1: Optional[float] = NotProvided,
        adam_beta2: Optional[float] = NotProvided,
        grad_clip: Optional[int] = NotProvided,
        num_steps_sampled_before_learning_starts: Optional[int] = NotProvided,
        tau: Optional[float] = NotProvided,
        num_atoms: Optional[int] = NotProvided,
        v_min: Optional[float] = NotProvided,
        v_max: Optional[float] = NotProvided,
        noisy: Optional[bool] = NotProvided,
        sigma0: Optional[float] = NotProvided,
        dueling: Optional[bool] = NotProvided,
        hiddens: Optional[int] = NotProvided,
        double_q: Optional[bool] = NotProvided,
        n_step: Optional[int] = NotProvided,
        before_learn_on_batch: Callable[
            [Type[MultiAgentBatch], List[Type[Policy]], Type[int]],
            Type[MultiAgentBatch],
        ] = NotProvided,
        training_intensity: Optional[float] = NotProvided,
        td_error_loss_fn: Optional[str] = NotProvided,
        categorical_distribution_temperature: Optional[float] = NotProvided,
        **kwargs,
    ) -> "DQNAlgoConfig":
        """Sets the training related configuration.

        Args:

            ------------------> SimpleQ

            target_network_update_freq: Update the target network every
                `target_network_update_freq` sample steps.
            replay_buffer_config: Replay buffer config.
                Examples:
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000,
                "replay_sequence_length": 1,
                }
                - OR -
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                "replay_sequence_length": 1,
                }
                - Where -
                prioritized_replay_alpha: Alpha parameter controls the degree of
                prioritization in the buffer. In other words, when a buffer sample has
                a higher temporal-difference error, with how much more probability
                should it drawn to use to update the parametrized Q-network. 0.0
                corresponds to uniform probability. Setting much above 1.0 may quickly
                result as the sampling distribution could become heavily “pointy” with
                low entropy.
                prioritized_replay_beta: Beta parameter controls the degree of
                importance sampling which suppresses the influence of gradient updates
                from samples that have higher probability of being sampled via alpha
                parameter and the temporal-difference error.
                prioritized_replay_eps: Epsilon parameter sets the baseline probability
                for sampling so that when the temporal-difference error of a sample is
                zero, there is still a chance of drawing the sample.
            store_buffer_in_checkpoints: Set this to True, if you want the contents of
                your buffer(s) to be stored in any saved checkpoints as well.
                Warnings will be created if:
                - This is True AND restoring from a checkpoint that contains no buffer
                data.
                - This is False AND restoring from a checkpoint that does contain
                buffer data.
            lr_schedule: Learning rate schedule. In the format of [[timestep, value],
                [timestep, value], ...]. A schedule should normally start from
                timestep 0.
            grad_clip: If not None, clip gradients during optimization at this value.
            num_steps_sampled_before_learning_starts: Number of timesteps to collect
                from rollout workers before we start sampling from replay buffers for
                learning. Whether we count this in agent steps  or environment steps
                depends on config.multi_agent(count_steps_by=..).
            tau: Update the target by \tau * policy + (1-\tau) * target_policy.

            -------------------> DQN

            num_atoms: Number of atoms for representing the distribution of return.
                When this is greater than 1, distributional Q-learning is used.
            v_min: Minimum value estimation
            v_max: Maximum value estimation
            noisy: Whether to use noisy network to aid exploration. This adds parametric
                noise to the model weights.
            sigma0: Control the initial parameter noise for noisy nets.
            dueling: Whether to use dueling DQN.
            hiddens: Dense-layer setup for each the advantage branch and the value
                branch
            double_q: Whether to use double DQN.
            n_step: N-step for Q-learning.
            before_learn_on_batch: Callback to run before learning on a multi-agent
                batch of experiences.
            training_intensity: The intensity with which to update the model (vs
                collecting samples from the env).
                If None, uses "natural" values of:
                `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
                `num_envs_per_worker`).
                If not None, will make sure that the ratio between timesteps inserted
                into and sampled from the buffer matches the given values.
                Example:
                training_intensity=1000.0
                train_batch_size=250
                rollout_fragment_length=1
                num_workers=1 (or 0)
                num_envs_per_worker=1
                -> natural value = 250 / 1 = 250.0
                -> will make sure that replay+train op will be executed 4x asoften as
                rollout+insert op (4 * 250 = 1000).
                See: rllib/algorithms/dqn/dqn.py::calculate_rr_weights for further
                details.
            td_error_loss_fn: "huber" or "mse". loss function for calculating TD error
                when num_atoms is 1. Note that if num_atoms is > 1, this parameter
                is simply ignored, and softmax cross entropy loss will be used.
            categorical_distribution_temperature: Set the temperature parameter used
                by Categorical action distribution. A valid temperature is in the range
                of [0, 1]. Note that this mostly affects evaluation since TD error uses
                argmax for return calculation.

        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        # --------------> SimpleQ
        if target_network_update_freq is not NotProvided:
            self.target_network_update_freq = target_network_update_freq
        if replay_buffer_config is not NotProvided:
            # Override entire `replay_buffer_config` if `type` key changes.
            # Update, if `type` key remains the same or is not specified.
            new_replay_buffer_config = deep_update(
                {"replay_buffer_config": self.replay_buffer_config},
                {"replay_buffer_config": replay_buffer_config},
                False,
                ["replay_buffer_config"],
                ["replay_buffer_config"],
            )
            self.replay_buffer_config = new_replay_buffer_config["replay_buffer_config"]
        if store_buffer_in_checkpoints is not NotProvided:
            self.store_buffer_in_checkpoints = store_buffer_in_checkpoints
        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if adam_beta1 is not NotProvided:
            self.adam_beta1 = adam_beta1
        if adam_beta2 is not NotProvided:
            self.adam_beta2 = adam_beta2
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if num_steps_sampled_before_learning_starts is not NotProvided:
            self.num_steps_sampled_before_learning_starts = num_steps_sampled_before_learning_starts
        if tau is not NotProvided:
            self.tau = tau

        # --------------> DQN
        if num_atoms is not NotProvided:
            self.num_atoms = num_atoms
        if v_min is not NotProvided:
            self.v_min = v_min
        if v_max is not NotProvided:
            self.v_max = v_max
        if noisy is not NotProvided:
            self.noisy = noisy
        if sigma0 is not NotProvided:
            self.sigma0 = sigma0
        if dueling is not NotProvided:
            self.dueling = dueling
        if hiddens is not NotProvided:
            self.hiddens = hiddens
        if double_q is not NotProvided:
            self.double_q = double_q
        if n_step is not NotProvided:
            self.n_step = n_step
        if before_learn_on_batch is not NotProvided:
            self.before_learn_on_batch = before_learn_on_batch
        if training_intensity is not NotProvided:
            self.training_intensity = training_intensity
        if td_error_loss_fn is not NotProvided:
            self.td_error_loss_fn = td_error_loss_fn
        if categorical_distribution_temperature is not NotProvided:
            self.categorical_distribution_temperature = categorical_distribution_temperature

        return self

    @override(SimpleQConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()

        if self.exploration_config["type"] == "ParameterNoise":
            if self.batch_mode != "complete_episodes":
                raise ValueError(
                    "ParameterNoise Exploration requires `batch_mode` to be "
                    "'complete_episodes'. Try setting `config.rollouts("
                    "batch_mode='complete_episodes')`."
                )

        if not self.in_evaluation:
            validate_buffer_config(self)

        if self.td_error_loss_fn not in ["huber", "mse"]:
            raise ValueError("`td_error_loss_fn` must be 'huber' or 'mse'!")

        # Check rollout_fragment_length to be compatible with n_step.
        if (
            not self.in_evaluation
            and self.rollout_fragment_length != "auto"
            and self.rollout_fragment_length < self.n_step
        ):
            raise ValueError(
                f"Your `rollout_fragment_length` ({self.rollout_fragment_length}) is "
                f"smaller than `n_step` ({self.n_step})! "
                f"Try setting config.rollouts(rollout_fragment_length={self.n_step})."
            )

        if self.exploration_config["type"] == "ParameterNoise":
            if self.batch_mode != "complete_episodes":
                raise ValueError(
                    "ParameterNoise Exploration requires `batch_mode` to be "
                    "'complete_episodes'. Try setting `config.rollouts("
                    "batch_mode='complete_episodes')`."
                )
            if self.noisy:
                raise ValueError(
                    "ParameterNoise Exploration and `noisy` network cannot be"
                    " used at the same time!"
                )

    @override(AlgorithmConfig)
    def get_rollout_fragment_length(self, worker_index: int = 0) -> int:
        if self.rollout_fragment_length == "auto":
            return self.n_step
        else:
            return self.rollout_fragment_length

def calculate_rr_weights(config: AlgorithmConfig) -> List[float]:
    """Calculate the round robin weights for the rollout and train steps"""
    if not config["training_intensity"]:
        return [1, 1]

    # Calculate the "native ratio" as:
    # [train-batch-size] / [size of env-rolled-out sampled data]
    # This is to set freshly rollout-collected data in relation to
    # the data we pull from the replay buffer (which also contains old
    # samples).
    native_ratio = config["train_batch_size"] / (
        config.get_rollout_fragment_length()
        * config["num_envs_per_worker"]
        # Add one to workers because the local
        # worker usually collects experiences as well, and we avoid division by zero.
        * max(config["num_workers"] + 1, 1)
    )

    # Training intensity is specified in terms of
    # (steps_replayed / steps_sampled), so adjust for the native ratio.
    sample_and_train_weight = config["training_intensity"] / native_ratio
    if sample_and_train_weight < 1:
        return [int(np.round(1 / sample_and_train_weight)), 1]
    else:
        return [1, int(np.round(sample_and_train_weight))]

class DQN(Algorithm):
    def __init__(
        self,
        config: Optional[AlgorithmConfig] = None,
        env=None,  # deprecated arg
        logger_creator: Optional[Callable[[], Logger]] = None,
        **kwargs,
    ):
        super().__init__(config, env, logger_creator, **kwargs)

        # store custom data
        self.data_dict = {}
        self.subtask = 0

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return DQNAlgoConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(cls, config: AlgorithmConfig):
        if config["framework"] == "torch":
            raise NotImplementedError(
                "Torch framework is not supported for this algorithm class.")
        return DQNPolicy

    def before_subtask(self, mode):
        self.workers.local_worker().async_env.first_reset_done = False
        self.workers.local_worker().async_env.try_restart()

        # self.workers.local_worker().async_env.restart_at()
        # self.workers.foreach_env(lambda e: e.restart_at())

        # self.workers.reset()

        # self.workers.local_worker().async_env.new_obs = None
        # self.workers.local_worker().async_env.cur_rewards = None
        # self.workers.local_worker().async_env.cur_terminateds = None
        # self.workers.local_worker().async_env.cur_truncateds = None
        # self.workers.local_worker().async_env.cur_infos = None

        # rollout_worker.sampler.env_runner_v2.vector_env.new_obs = None
        # rollout_worker.sampler.env_runner_v2.vector_env.cur_rewards = None
        # rollout_worker.sampler.env_runner_v2.vector_env.cur_terminateds = None
        # rollout_worker.sampler.env_runner_v2.vector_env.cur_truncateds = None
        # rollout_worker.sampler.env_runner_v2.vector_env.cur_infos = None
        # rollout_worker.sampler.env_runner_v2.vector_env.first_reset_done = False

        # self.workers.local_worker()._needs_initial_reset = True
        # self.workers.local_worker()._episode = None

        # self.workers.local_worker().sample(force_reset=True)
        # self.workers.foreach_worker(lambda w: w.sample(force_reset=True))
        # self.workers.foreach_env(lambda e: e.reset())

        # ---

        # print(self._timers)
        # print(self._counters)
        # print(self._episode_history)
        # print(self._episodes_to_be_collected)

        # print(self.evaluation_metrics)

        # ---

        # self._timers.clear()
        # self._counters.clear()
        # self._episode_history.clear()
        # self._episodes_to_be_collected.clear()

        # self._iteration = 0
        # self._time_total = 0.0
        # self._timesteps_total = None
        # self._episodes_total = None
        # self._time_since_restore = 0.0
        # self._timesteps_since_restore = 0
        # self._iterations_since_restore = 0
        # self._last_result = None
        # self._restored = False

    def after_subtask(self, mode):
        self.subtask += 1
    
    @override(Algorithm)
    def evaluate(self, track, duration_fn=None):
        self.get_policy()._before_evaluate(track)
        if self.config.fuck_it:
            step_results = super().evaluate(duration_fn)
        else:
            if self.config.evaluation_duration_unit == 'episodes':
                frequency = 100
                count = max(1, self.config.evaluation_duration // frequency)
                for i in range(self.config.evaluation_duration):
                    if self.config.debug or i % count == 0:
                        Logger().info(f'{i + 1}/{self.config.evaluation_duration} {self.config.evaluation_duration_unit}')
                    snapshot = len(self._episode_history)
                    while snapshot == len(self._episode_history):
                        step_results = super().evaluate(duration_fn)
            if self.config.evaluation_duration_unit == 'timesteps':
                frequency = 100
                count = max(1, self.config.evaluation_duration // frequency)
                for i in range(self.config.evaluation_duration):
                    print('timers:', self._timers)
                    print('counters:', self._counters)
                    if self.config.debug or i % count == 0:
                        Logger().info(f'{i + 1}/{self.config.evaluation_duration} {self.config.evaluation_duration_unit}')
                    if i == self.config.evaluation_duration - 1:
                        self.workers.foreach_env(lambda e: e.override())
                    step_results = super().evaluate(duration_fn)
        self.cache_data('eval')
        self.get_policy()._after_evaluate()

        return {}
    
    def _evaluate(self, duration_fn=None):
        """Evaluates current policy under `evaluation_config` settings.

        Args:
            duration_fn: An optional callable taking the already run
                num episodes as only arg and returning the number of
                episodes left to run. It's used to find out whether
                evaluation should continue.
        """
        # Call the `_before_evaluate` hook.
        self._before_evaluate()

        self.callbacks.on_evaluate_start(algorithm=self)

        if (
            self.evaluation_workers is None
            and self.workers.local_worker().input_reader is None
        ):
            raise ValueError(
                "Cannot evaluate w/o an evaluation worker set in "
                "the Algorithm or w/o an env on the local worker!\n"
                "Try one of the following:\n1) Set "
                "`evaluation_interval` >= 0 to force creating a "
                "separate evaluation worker set.\n2) Set "
                "`create_env_on_driver=True` to force the local "
                "(non-eval) worker to have an environment to "
                "evaluate on."
            )

        # How many episodes/timesteps do we need to run?
        # In "auto" mode (only for parallel eval + training): Run as long
        # as training lasts.
        unit = self.config.evaluation_duration_unit
        eval_cfg = self.evaluation_config
        rollout = eval_cfg.rollout_fragment_length
        num_envs = eval_cfg.num_envs_per_worker
        auto = self.config.evaluation_duration == "auto"
        duration = (
            self.config.evaluation_duration
            if not auto
            else (self.config.evaluation_num_workers or 1)
            * (1 if unit == "episodes" else rollout)
        )
        agent_steps_this_iter = 0
        env_steps_this_iter = 0
        
        # Default done-function returns True, whenever num episodes
        # have been completed.
        if duration_fn is None:

            def duration_fn(num_units_done):
                return duration - num_units_done

        #print(f"Evaluating current state of {self} for {duration} {unit}.")

        metrics = None
        all_batches = []
        # No evaluation worker set ->
        # Do evaluation using the local worker. Expect error due to the
        # local worker not having an env.
        if self.evaluation_workers is None:
            # If unit=episodes -> Run n times `sample()` (each sample
            # produces exactly 1 episode).
            # If unit=ts -> Run 1 `sample()` b/c the
            # `rollout_fragment_length` is exactly the desired ts.
            iters = duration if unit == "episodes" else 1

            for _ in range(iters):
                batch = self.workers.local_worker().sample()
                agent_steps_this_iter += batch.agent_steps()
                env_steps_this_iter += batch.env_steps()
                if self.reward_estimators:
                    all_batches.append(batch)
            metrics = collect_metrics(
                self.workers,
                keep_custom_metrics=eval_cfg.keep_per_episode_custom_metrics,
                timeout_seconds=eval_cfg.metrics_episode_collection_timeout_s,
            )

        # Evaluation worker set only has local worker.
        elif self.evaluation_workers.num_remote_workers() == 0:
            # If unit=episodes -> Run n times `sample()` (each sample
            # produces exactly 1 episode).
            # If unit=ts -> Run 1 `sample()` b/c the
            # `rollout_fragment_length` is exactly the desired ts.
            iters = duration if unit == "episodes" else 1
            for _ in range(iters):
                batch = self.evaluation_workers.local_worker().sample()
                agent_steps_this_iter += batch.agent_steps()
                env_steps_this_iter += batch.env_steps()
                if self.reward_estimators:
                    all_batches.append(batch)
        else:
            # Can't find a good way to run this evaluation.
            # Wait for next iteration.
            pass

        if metrics is None:
            metrics = collect_metrics(
                self.evaluation_workers,
                keep_custom_metrics=self.config.keep_per_episode_custom_metrics,
                timeout_seconds=eval_cfg.metrics_episode_collection_timeout_s,
            )

        # TODO: Don't dump sampler results into top-level.
        if not self.config.custom_evaluation_function:
            metrics = dict({"sampler_results": metrics}, **metrics)

        metrics[NUM_AGENT_STEPS_SAMPLED_THIS_ITER] = agent_steps_this_iter
        metrics[NUM_ENV_STEPS_SAMPLED_THIS_ITER] = env_steps_this_iter
        # TODO: Remove this key at some point. Here for backward compatibility.
        metrics["timesteps_this_iter"] = env_steps_this_iter

        # Compute off-policy estimates
        estimates = defaultdict(list)
        # for each batch run the estimator's fwd pass
        for name, estimator in self.reward_estimators.items():
            for batch in all_batches:
                estimate_result = estimator.estimate(
                    batch,
                    split_batch_by_episode=self.config.ope_split_batch_by_episode,
                )
                estimates[name].append(estimate_result)

        # collate estimates from all batches
        if estimates:
            metrics["off_policy_estimator"] = {}
            for name, estimate_list in estimates.items():
                avg_estimate = tree.map_structure(
                    lambda *x: np.mean(x, axis=0), *estimate_list
                )
                metrics["off_policy_estimator"][name] = avg_estimate

        # Evaluation does not run for every step.
        # Save evaluation metrics on Algorithm, so it can be attached to
        # subsequent step results as latest evaluation result.
        self.evaluation_metrics = {"evaluation": metrics}

        # Trigger `on_evaluate_end` callback.
        self.callbacks.on_evaluate_end(
            algorithm=self, evaluation_metrics=self.evaluation_metrics
        )

        # Also return the results here for convenience.
        return self.evaluation_metrics

    @override(Algorithm)
    def train(self, track):
        self.get_policy()._before_train(track)
        if self.config.fuck_it:
            step_results = super().train()
            self.gather_data('train', step_results)
        else:
            if self.config.training_duration_unit == 'episodes':
                frequency = 100
                count = max(1, self.config.training_duration // frequency)
                for i in range(self.config.training_duration):
                    if self.config.debug or i % count == 0:
                        Logger().info(f'{i + 1}/{self.config.training_duration} {self.config.training_duration_unit}')
                    snapshot = len(self._episode_history)
                    while snapshot == len(self._episode_history):
                        step_results = super().train()
                        self.gather_data('train', step_results)
            if self.config.training_duration_unit == 'timesteps':
                frequency = 100
                count = max(1, self.config.training_duration // frequency)
                for i in range(self.config.training_duration):
                    print('timers:', self._timers)
                    print('counters:', self._counters)
                    if self.config.debug or i % count == 0:
                        Logger().info(f'{i + 1}/{self.config.training_duration} {self.config.training_duration_unit}')
                    if i == self.config.training_duration - 1:
                        self.workers.foreach_env(lambda e: e.override())
                    step_results = super().train()
                    self.gather_data('train', step_results)
        self.cache_data('train')
        self.get_policy()._after_train()

        return {}
    
    @override(Algorithm)
    def step(self) -> ResultDict:
        """Implements the main `Algorithm.train()` logic.

        Takes n attempts to perform a single training step. Thereby
        catches RayErrors resulting from worker failures. After n attempts,
        fails gracefully.

        Override this method in your Algorithm sub-classes if you would like to
        handle worker failures yourself.
        Otherwise, override only `training_step()` to implement the core
        algorithm logic.

        Returns:
            The results dict with stats/infos on sampling, training,
            and - if required - evaluation.
        """

        # Results dict for training (and if appolicable: evaluation).
        results: ResultDict = {}

        results, train_iter_ctx = self._run_one_training_iteration()

        if hasattr(self, "workers") and isinstance(self.workers, WorkerSet):
            # Sync filters on workers.
            self._sync_filters_if_needed(
                central_worker=self.workers.local_worker(),
                workers=self.workers,
                config=self.config,
            )
            # TODO (avnishn): Remove the execution plan API by q1 2023
            # Collect worker metrics and add combine them with `results`.
            if self.config._disable_execution_plan_api:
                episodes_this_iter = collect_episodes(
                    self.workers,
                    self._remote_worker_ids_for_metrics(),
                    timeout_seconds=self.config.metrics_episode_collection_timeout_s,
                )
                results = self._compile_iteration_results(
                    episodes_this_iter=episodes_this_iter,
                    step_ctx=train_iter_ctx,
                    iteration_results=results,
                )

        return results

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        """DQN training iteration function.

        - Sample n MultiAgentBatches from n workers synchronously.
        - Store new samples in the replay buffer.
        - Sample one training MultiAgentBatch from the replay buffer.
        - Update remote workers' new policy weights.
        - Update target network every `target_network_update_freq` sample steps.
        - Return all collected metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """
        train_results = {}

        # We alternate between storing new samples and sampling and training
        store_weight, sample_and_train_weight = calculate_rr_weights(self.config)

        for _ in range(store_weight):
            # Sample (MultiAgentBatch) from workers.
            with self._timers[SAMPLE_TIMER]:
                new_sample_batch = synchronous_parallel_sample(
                    worker_set=self.workers, concat=True
                )

            # Update counters
            self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

            # Store new samples in replay buffer.
            self.local_replay_buffer.add(new_sample_batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        # Update target network every `target_network_update_freq` sample steps.
        cur_ts = self._counters[
            NUM_AGENT_STEPS_SAMPLED
            if self.config.count_steps_by == "agent_steps"
            else NUM_ENV_STEPS_SAMPLED
        ]

        if cur_ts > self.config.num_steps_sampled_before_learning_starts:
            for _ in range(sample_and_train_weight):
                # Sample training batch (MultiAgentBatch) from replay buffer.
                train_batch = sample_min_n_steps_from_buffer(
                    self.local_replay_buffer,
                    self.config.train_batch_size,
                    count_by_agent_steps=self.config.count_steps_by == "agent_steps",
                )

                # Postprocess batch before we learn on it
                post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
                train_batch = post_fn(train_batch, self.workers, self.config)

                # cur_obs_mean = np.mean(new_sample_batch['default_policy']['obs'].mean())
                # mean_obs = np.mean(np.mean(np.squeeze(train_batch['default_policy']['obs'], axis=(3,)), axis=1), axis=1)
                # cur_obs_index = np.argwhere(mean_obs==cur_obs_mean)

                if self.config.get("simple_optimizer") is True:
                    train_results = train_one_step(self, train_batch)
                else:
                    train_results = multi_gpu_train_one_step(self, train_batch)

                # get q values for current training batch
                self.update_data('train_offline_q_values', self.get_policy().q_t)

                # Update replay buffer priorities.
                update_priorities_in_replay_buffer(
                    self.local_replay_buffer,
                    self.config,
                    train_batch,
                    train_results,
                )

                if self.config.get("target_network") is True:
                    last_update = self._counters[LAST_TARGET_UPDATE_TS]
                    if cur_ts - last_update >= self.config.target_network_update_freq:
                        to_update = self.workers.local_worker().get_policies_to_train()
                        self.workers.local_worker().foreach_policy_to_train(
                            lambda p, pid: pid in to_update and p.update_target()
                        )
                        self._counters[NUM_TARGET_UPDATES] += 1
                        self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

                # Update weights and global_vars - after learning on the local worker -
                # on all remote workers.
                with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                    self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results

    def update_data(self, key, value):
        try: self.data_dict[key].append(value)
        except: self.data_dict[key] = [value]

    def gather_data(self, caller, step_results):
        # pprint(step_results)
        sample_time = np.divide(step_results['timers']['sample_time_ms'], 1e+03, dtype=np.float32)
        train_iter_time = np.divide(step_results['timers']['training_iteration_time_ms'], 1e+03, dtype=np.float32)

        # empty if policy is in sample-only mode
        mean_q, mean_td_error, loss = np.nan, np.nan, np.nan
        if step_results['info']['learner']:
            learner_stats = step_results['info']['learner']['default_policy']['learner_stats']

            mean_q        = np.array(learner_stats['mean_q'], dtype=np.float32)
            mean_td_error = np.array(learner_stats['mean_td_error'], dtype=np.float32)
            loss          = np.array(learner_stats['loss'], dtype=np.float32)

        self.update_data(f'{caller}_q_values', mean_q)
        self.update_data(f'{caller}_td_error', mean_td_error)
        self.update_data(f'{caller}_loss', loss)
        self.update_data(f'{caller}_sample_time', sample_time)
        self.update_data(f'{caller}_iteration_time', train_iter_time)

    def cache_data(self, mode):
        for entry in self.data_dict:
            if entry.find(mode) == -1: continue

            if entry not in Cache().object_registry:
                Cache().register_object(entry, None, category='data')

            current_object = Cache().object_registry[entry][0]
            try: updated_object = np.stack([current_object, np.array(self.data_dict[entry])])
            except: updated_object = np.array(self.data_dict[entry])
            Cache().update_object(entry, updated_object)

            self.data_dict[entry].clear()
