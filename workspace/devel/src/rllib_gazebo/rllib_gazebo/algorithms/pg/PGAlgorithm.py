import os
import time
import numpy as np

from pprint import pprint
from typing import List, Optional, Type, Union, Callable

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided

from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step, multi_gpu_train_one_step

from ray.rllib.policy.policy import Policy

from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration import StochasticSampling
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
)

from rllib_gazebo.algorithms.pg.PGPolicy import PGTFPolicy
from rllib_gazebo.algorithms.utils.exploration.EpsGreedy import EpsilonGreedy

from rllib_gazebo.utils.Evaluation import *
from rllib_gazebo.utils.ArgsConfig import LearnerArgs
from rllib_gazebo.utils.Exchanging import Exchanger
from rllib_gazebo.utils.Logging import Logger
from rllib_gazebo.utils.Caching import Cache


class PGConfig(AlgorithmConfig):
    """Defines a configuration class from which a PG Algorithm can be built."""

    def __init__(self, algo_class=None):
        """Initializes a PGConfig instance."""
        super().__init__(algo_class=algo_class or PG)

        largs = LearnerArgs().args

        self.fuck_it = largs.fuck_it
        self.debug = largs.debug
        self.rollout_fragment_length = "auto"

        exp_id = largs.exp_id
        root_dir = largs.root_dir
        self.results_dir = os.path.join(root_dir, exp_id, 'debug')

        self.training_duration = largs.training_duration
        self.training_duration_unit = largs.training_duration_unit

         # ---> PG defaults
        self.lr = largs.lr        # default: 4e-4
        self.lr_schedule = largs.lr_schedule

        self.train_batch_size = largs.train_batch_size
        self._disable_preprocessor_api = True

         # --------------> `exploration()`
        self.explore = False
        self.exploration_config = {}
        if largs.exploration is not None:
            self.explore = True
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
                self.exploration_config.update({
                    'type': EpsilonGreedy,
                    'initial_epsilon': largs.initial_epsilon,
                    'final_epsilon': largs.final_epsilon,
                    'warmup_timesteps': largs.warmup_timesteps,
                    'epsilon_timesteps': largs.epsilon_timesteps,
                    'epsilon_schedule': schedule,
                })
            elif largs.exploration == 'stochastic-sampling':
                self.exploration_config.update({
                    'type': StochasticSampling,
                    'random_timesteps': largs.random_timesteps
                })

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        lr_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        **kwargs,
    ) -> "PGConfig":
        """Sets the training related configuration.

        Args:
            gamma: Float specifying the discount factor of the Markov Decision process.
            lr: The default learning rate.
            train_batch_size: Training batch size, if applicable.
            model: Arguments passed into the policy model. See models/catalog.py for a
                full list of the available model options.
            optimizer: Arguments to pass to the policy optimizer.
            lr_schedule: Learning rate schedule. In the format of
                [[timestep, lr-value], [timestep, lr-value], ...]
                Intermediary timesteps will be assigned to interpolated learning rate
                values. A schedule should normally start from timestep 0.

        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)
        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule

        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()

        # Synchronous sampling, on-policy PG algo -> Check mismatches between
        # `rollout_fragment_length` and `train_batch_size` to avoid user confusion.
        self.validate_train_batch_size_vs_rollout_fragment_length()

class PG(Algorithm):

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
        return PGConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            raise NotImplementedError(
                "Torch framework is not supported for this algorithm class.")
        return PGTFPolicy

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
    def training_step(self) -> ResultDict:
        """Default single iteration logic of an algorithm.

        - Collect on-policy samples (SampleBatches) in parallel using the
          Algorithm's EnvRunners (@ray.remote).
        - Concatenate collected SampleBatches into one train batch.
        - Note that we may have more than one policy in the multi-agent case:
          Call the different policies' `learn_on_batch` (simple optimizer) OR
          `load_batch_into_buffer` + `learn_on_loaded_batch` (multi-GPU
          optimizer) methods to calculate loss and update the model(s).
        - Return all collected metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """
        # Collect SampleBatches from sample workers until we have a full batch.
        with self._timers[SAMPLE_TIMER]:
            if self.config.count_steps_by == "agent_steps":
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_agent_steps=self.config.train_batch_size,
                )
            else:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers, max_env_steps=self.config.train_batch_size
                )
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Only train if train_batch is not empty.
        # In an extreme situation, all rollout workers die during the
        # synchronous_parallel_sample() call above.
        # In which case, we should skip training, wait a little bit, then probe again.
        train_results = {}
        if train_batch.agent_steps() > 0:
            # Use simple optimizer (only for multi-agent or tf-eager; all other
            # cases should use the multi-GPU optimizer, even if only using 1 GPU).
            if self.config._enable_new_api_stack:
                is_module_trainable = self.workers.local_worker().is_policy_to_train
                self.learner_group.set_is_module_trainable(is_module_trainable)
                train_results = self.learner_group.update(train_batch)
            elif self.config.get("simple_optimizer") is True:
                train_results = train_one_step(self, train_batch)
            else:
                train_results = multi_gpu_train_one_step(self, train_batch)
        else:
            # Wait 1 sec before probing again via weight syncing.
            time.sleep(1)

        # Update weights and global_vars - after learning on the local worker - on all
        # remote workers (only those policies that were actually trained).
        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            from_worker_or_trainer = None
            if self.config._enable_new_api_stack:
                from_worker_or_trainer = self.learner_group
            self.workers.sync_weights(
                from_worker_or_learner_group=from_worker_or_trainer,
                policies=list(train_results.keys()),
                global_vars=global_vars,
            )

        return train_results

    def update_data(self, key, value):
        try: self.data_dict[key].append(value)
        except: self.data_dict[key] = [value]

    def gather_data(self, caller, step_results):
        # pprint(step_results)
        sample_time = np.divide(step_results['timers']['sample_time_ms'], 1e+03, dtype=np.float32)
        train_iter_time = np.divide(step_results['timers']['training_iteration_time_ms'], 1e+03, dtype=np.float32)

        # empty if policy is in sample-only mode
        policy_loss = np.nan
        if step_results['info']['learner']:
            learner_stats = step_results['info']['learner']['default_policy']['learner_stats']

            policy_loss = np.array(learner_stats['policy_loss'], dtype=np.float32)

        self.update_data(f'{caller}_policy_loss', policy_loss)
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
