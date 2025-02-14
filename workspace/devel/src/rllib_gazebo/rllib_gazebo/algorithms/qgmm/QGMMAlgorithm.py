import os
import numpy as np

from pprint import pprint
from typing import List, Optional, Callable

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.simple_q.simple_q import SimpleQConfig

from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step, multi_gpu_train_one_step

from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.exploration import ParameterNoise, StochasticSampling
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
)
from ray.rllib.utils.replay_buffers.utils import (
    sample_min_n_steps_from_buffer,
    update_priorities_in_replay_buffer,
    validate_buffer_config,
)
from ray.rllib.utils.schedules import PiecewiseSchedule

from rllib_gazebo.algorithms.qgmm.QGMMPolicy import QGMMPolicy
from rllib_gazebo.algorithms.utils.exploration.EpsGreedy import EpsilonGreedy

from rllib_gazebo.utils.Evaluation import *
from rllib_gazebo.utils.ArgsConfig import LearnerArgs
from rllib_gazebo.utils.Exchanging import Exchanger
from rllib_gazebo.utils.Logging import Logger
from rllib_gazebo.utils.Caching import Cache


class QGMMAlgoConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        ''' QGMM AlgoConfig '''
        super().__init__(algo_class=algo_class or QGMM)

        largs = LearnerArgs().args

        self.fuck_it = largs.fuck_it
        self.debug = largs.debug
        self.rollout_fragment_length = "auto"

        exp_id = largs.exp_id
        root_dir = largs.root_dir
        self.results_dir = os.path.join(root_dir, exp_id, 'debug')

        self.training_duration = largs.training_duration
        self.training_duration_unit = largs.training_duration_unit

        self.training_intensity = largs.training_intensity

        self.num_steps_sampled_before_learning_starts = largs.num_steps_sampled_before_learning_starts
        self.train_batch_size = largs.train_batch_size

        # --------------> `training()`
        self.grad_clip = 40.0
        self.grad_clip_by = "global_norm"

        # --------------> `exploration()`
        self.explore = False
        self.exploration_config = {}
        if largs.exploration == 'eps-greedy':
            schedule = None
            interpolations = {
                'poly': lambda left, right, alpha: np.add(right, np.multiply(np.subtract(left, right), np.power(1 - alpha, largs.schedule_power))),
                'exp': lambda left, right, alpha: np.add(right, np.multiply(np.subtract(left, right), np.power(largs.schedule_decay, alpha))),
                'poly_inv': lambda left, right, alpha: np.add(left, np.multiply(np.subtract(right, left), np.power(alpha, largs.schedule_power))),
                'exp_inv': lambda left, right, alpha: np.add(left, np.multiply(np.subtract(right, left), np.power(largs.schedule_decay, (1 - alpha)))),
            }
            if largs.interpolation in interpolations:
                PiecewiseSchedule(framework='tf2', interpolation=interpolations[largs.interpolation], outside_value=largs.final_epsilon,
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
        elif largs.exploration == 'stochastic-sampling':
            self.explore = True
            self.exploration_config.update({
                'type': StochasticSampling,
                'random_timesteps': largs.random_timesteps
            })

        # `reporting()`
        self.min_time_s_per_iteration = None
        self.min_train_timesteps_per_iteration = None
        self.min_sample_timesteps_per_iteration = None
        self.store_buffer_in_checkpoints = False

        # --------------> ReplayBuffer
        self.replay_buffer_choice = largs.replay_buffer
        self.replay_buffer_config = {
            # The number of continuous environment steps to replay at once.
            # This may be set to greater than 1 to support recurrent models.
            'type': 'ReservoirReplayBuffer',
            'replay_sequence_length': largs.replay_sequence_length,
            'capacity': largs.capacity,
            # Whether to compute priorities on workers.
            'worker_side_prioritization': False,
            'storage_unit': 'timesteps',  # 'timesteps', 'sequences', 'episodes'
        }
        if self.replay_buffer_choice == 'prioritized':
            # Replay buffer configuration.
            self.replay_buffer_config.update({
                "type": "MultiAgentPrioritizedReplayBuffer",
                "prioritized_replay": DEPRECATED_VALUE,
                "prioritized_replay_alpha": largs.per_alpha,
                # Beta parameter for sampling from prioritized replay buffer.
                "prioritized_replay_beta": largs.per_beta,
                # Epsilon to add to the TD errors when updating priorities.
                "prioritized_replay_eps": largs.per_eps,
            })

    @override(AlgorithmConfig)
    def training(self,
        replay_buffer_config: Optional[dict] = NotProvided,
        store_buffer_in_checkpoints: Optional[bool] = NotProvided,
        num_steps_sampled_before_learning_starts: Optional[int] = NotProvided,
        training_intensity: Optional[float] = NotProvided,
        **kwargs,
    ) -> "QGMMAlgoConfig":
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)
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
        if num_steps_sampled_before_learning_starts is not NotProvided:
            self.num_steps_sampled_before_learning_starts = (
                num_steps_sampled_before_learning_starts
            )
        if training_intensity is not NotProvided:
            self.training_intensity = training_intensity

        return self

    @override(SimpleQConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()

        if not self.in_evaluation:
            validate_buffer_config(self)

        if self.exploration_config["type"] == "ParameterNoise":
            if self.batch_mode != "complete_episodes":
                raise ValueError(
                    "ParameterNoise Exploration requires `batch_mode` to be "
                    "'complete_episodes'. Try setting `config.rollouts("
                    "batch_mode='complete_episodes')`."
                )

    @override(AlgorithmConfig)
    def get_rollout_fragment_length(self, worker_index: int = 0) -> int:
        if self.rollout_fragment_length == "auto":
            return 1

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

class QGMM(Algorithm):
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
        return QGMMAlgoConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(cls, config: AlgorithmConfig):
        if config["framework"] == "torch":
            raise NotImplementedError(
                "Torch framework is not supported for this algorithm class.")
        return QGMMPolicy

    def before_subtask(self, mode):
        self.workers.local_worker().async_env.first_reset_done = False
        self.workers.local_worker().async_env.try_restart()

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
                    if self.config.debug or i % count == 0:
                        Logger().info(f'{i + 1}/{self.config.evaluation_duration} {self.config.evaluation_duration_unit}')
                    if i == self.config.evaluation_duration - 1:
                        self.workers.foreach_env(lambda e: e.override())
                    step_results = super().evaluate(duration_fn)
        self.cache_data('eval')
        self.get_policy()._after_evaluate()

        return {}

    @override(Algorithm)
    def train(self, track):
        # self.serialized_obs = []
        # self.amount_to_serialize = 5000
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
    def training_step(self):
        """QGMM training iteration function.

        - Sample n MultiAgentBatches from n workers synchronously.
        - (Opt.) Store new samples in the replay buffer.
        - (Opt.) Sample one training MultiAgentBatch from the replay buffer.
        - Redirect one MultiAgentBatch to update remote workers' new policy weights.
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
            if self.config.replay_buffer_choice is not None:
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

        # INFO: Code snippet to serialize some obs. data for pre-training of GMMs!
        # print(f'Current sample timestep: {cur_ts}')
        # if self._counters[NUM_ENV_STEPS_SAMPLED] <= self.amount_to_serialize:
        #     reshaped_obs = np.squeeze(new_sample_batch['default_policy']['obs'], axis=(0,))
        #     self.serialized_obs.append(reshaped_obs)
        # if self._counters[NUM_ENV_STEPS_SAMPLED] == self.amount_to_serialize:
        #     self.serialized_obs = np.array(self.serialized_obs)
        #     save_path = os.path.join(self.config.results_dir, 'obs')
        #     np.save(save_path, self.serialized_obs)
        #     print(f'Saved array with dims: {self.serialized_obs.shape} to: {save_path}.npy')

        if cur_ts > self.config.num_steps_sampled_before_learning_starts:
            for _ in range(sample_and_train_weight):
                if self.config.replay_buffer_choice is not None: # buffer mode
                    # Sample training batch (MultiAgentBatch) from replay buffer.
                    train_batch = sample_min_n_steps_from_buffer(
                        self.local_replay_buffer,
                        self.config.train_batch_size,
                        count_by_agent_steps=self.config.count_steps_by == "agent_steps",
                    )
                else: # no buffer
                    train_batch = new_sample_batch

                # Postprocess batch before we learn on it
                post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
                train_batch = post_fn(train_batch, self.workers, self.config)

                if self.config.get("simple_optimizer") is True:
                    train_results = train_one_step(self, train_batch)
                else:
                    train_results = multi_gpu_train_one_step(self, train_batch)

                # get q values for current training batch
                self.update_data('train_offline_q_values', self.get_policy().q_cur)

                if self.config.replay_buffer_choice is not None:
                    # Update replay buffer priorities.
                    update_priorities_in_replay_buffer(
                        self.local_replay_buffer,
                        self.config,
                        train_batch,
                        train_results,
                    )

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

        learner_stats = step_results['info']['learner']['default_policy']

        q_losses   = np.array(learner_stats['q_losses'], dtype=np.float32)
        var_losses = np.array(learner_stats['var_losses'], dtype=np.float32)
        if var_losses.shape[0] == 1: # dirty hack to compensate LTM being headless
            var_losses = np.append(var_losses, np.nan)

        self.update_data(f'{caller}_q_losses', q_losses)
        self.update_data(f'{caller}_var_losses', var_losses)
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
