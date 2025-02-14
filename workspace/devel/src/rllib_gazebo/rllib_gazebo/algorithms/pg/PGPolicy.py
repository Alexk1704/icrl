import logging
import numpy as np
import tree

from gymnasium.spaces import Space
from typing import Dict, List, Type, Union, Optional

from ray.rllib.algorithms.pg.pg import PGConfig
from ray.rllib.algorithms.pg.utils import post_process_advantages

from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2

from ray.rllib.policy.eager_tf_policy import _disallow_var_creation
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import LearningRateSchedule

from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.annotations import (
    override,
    is_overridden,
)
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    TensorType,
)

from rllib_gazebo.utils.Evaluation import *
from rllib_gazebo.utils.Caching import Cache

tf1, tf, tfv = try_import_tf()
logger = logging.getLogger(__name__)


class PGTFPolicy(EagerTFPolicyV2):

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        config: AlgorithmConfigDict,
        **kwargs,
    ):
        existing_inputs = kwargs.get('existing_inputs', None)
        existing_model = kwargs.get('existing_model', None)

        # Enforce AlgorithmConfig for PG Policies.
        if isinstance(config, dict):
            config = PGConfig.from_dict(config)

        super().__init__( # Initialize base class.
            observation_space, action_space, config,
            existing_inputs=existing_inputs, existing_model=existing_model
        )

        # First thing first, enable eager execution if necessary.
        self.enable_eager_execution_if_necessary()

        # store custom data
        self.data_dict = {}
        self.is_training = None
        self.current_track_name = None

        LearningRateSchedule.__init__(self, config.lr, config.lr_schedule)

        # Note: this is a bit ugly, but loss and optimizer initialization must
        # happen after all the MixIns are initialized.
        self.maybe_initialize_optimizer_and_loss()

    def update_data(self, key, value):
        try: self.data_dict[key].append(value)
        except: self.data_dict[key] = [value]

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

    @override(EagerTFPolicyV2)
    def loss(
        self,
        model: Union[ModelV2, "tf.keras.Model"],
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """The basic policy gradients loss function.

        Calculates the vanilla policy gradient loss based on:
        L = -E[ log(pi(a|s)) * A]

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            Union[TensorType, List[TensorType]]: A single loss tensor or a list
                of loss tensors.
        """
        # Pass the training data through our model to get distribution parameters.
        dist_inputs, _ = model(train_batch)

        # Create an action distribution object.
        action_dist = dist_class(dist_inputs, model)

        # Calculate the vanilla PG loss based on:
        # L = -E[ log(pi(a|s)) * A]
        loss = -tf.reduce_mean(
            action_dist.logp(train_batch[SampleBatch.ACTIONS])
            * tf.cast(train_batch[Postprocessing.ADVANTAGES], dtype=tf.float32)
        )

        self.policy_loss = loss

        return loss

    @override(EagerTFPolicyV2)
    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[SampleBatch] = None,
        episode: Optional["Episode"] = None,
    ) -> SampleBatch:
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )
        return post_process_advantages(
            self, sample_batch, other_agent_batches, episode
        )

    @with_lock
    @override(EagerTFPolicyV2)
    def _compute_actions_helper(
        self,
        input_dict,
        state_batches,
        episodes,
        explore,
        timestep,
        _ray_trace_ctx=None,
    ):
        # Increase the tracing counter to make sure we don't re-trace too
        # often. If eager_tracing=True, this counter should only get
        # incremented during the @tf.function trace operations, never when
        # calling the already traced function after that.
        self._re_trace_counter += 1

        # Calculate RNN sequence lengths.
        if SampleBatch.SEQ_LENS in input_dict:
            seq_lens = input_dict[SampleBatch.SEQ_LENS]
        else:
            batch_size = tree.flatten(input_dict[SampleBatch.OBS])[0].shape[0]
            seq_lens = tf.ones(batch_size, dtype=tf.int32) if state_batches else None

        # Add default and custom fetches.
        extra_fetches = {}

        with tf.variable_creator_scope(_disallow_var_creation):

            if is_overridden(self.action_sampler_fn):
                actions, logp, dist_inputs, state_out = self.action_sampler_fn(
                    self.model,
                    input_dict[SampleBatch.OBS],
                    explore=explore,
                    timestep=timestep,
                    episodes=episodes,
                )
            else:
                if is_overridden(self.action_distribution_fn):
                    # Try new action_distribution_fn signature, supporting
                    # state_batches and seq_lens.
                    (
                        dist_inputs,
                        self.dist_class,
                        state_out,
                    ) = self.action_distribution_fn(
                        self.model,
                        obs_batch=input_dict[SampleBatch.OBS],
                        state_batches=state_batches,
                        seq_lens=seq_lens,
                        explore=explore,
                        timestep=timestep,
                        is_training=False,
                    )
                elif isinstance(self.model, tf.keras.Model):
                    if state_batches and "state_in_0" not in input_dict:
                        for i, s in enumerate(state_batches):
                            input_dict[f"state_in_{i}"] = s
                    self._lazy_tensor_dict(input_dict)
                    dist_inputs, state_out, extra_fetches = self.model(input_dict)
                else:
                    dist_inputs, state_out = self.model(
                        input_dict, state_batches, seq_lens
                    )

                action_dist = self.dist_class(dist_inputs, self.model)

                # Get the exploration action from the forward results.
                actions, logp, self.randomly_chosen = self.exploration.get_exploration_action(
                    action_distribution=action_dist,
                    timestep=timestep,
                    explore=explore,
                )

                # save the cause for the action based on the current obs (chosen randomly via exploration?)
                if Evaluator().get_entity(Evaluator().SAMPLE):
                    Evaluator().get_entity(Evaluator().SAMPLE).random = np.array(self.randomly_chosen, dtype=np.bool_)

        # Action-logp and action-prob.
        if logp is not None:
            extra_fetches[SampleBatch.ACTION_PROB] = tf.exp(logp)
            extra_fetches[SampleBatch.ACTION_LOGP] = logp
        # Action-dist inputs.
        if dist_inputs is not None:
            extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs
        # Custom extra fetches.
        extra_fetches.update(self.extra_action_out_fn())

        return actions, state_out, extra_fetches

    @override(EagerTFPolicyV2)
    def extra_learn_fetches_fn(self) -> Dict[str, TensorType]:
        return {
            "learner_stats": {"cur_lr": self.cur_lr},
        }

    @override(EagerTFPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        """Returns the calculated loss and learning rate in a stats dict.

        Args:
            policy: The Policy object.
            train_batch: The data used for training.

        Returns:
            Dict[str, TensorType]: The stats dict.
        """

        return {
            "policy_loss": self.policy_loss,
            "cur_lr": self.cur_lr,
        }

    def _before_train(self, track):
        self.is_training = True
        self.current_track_name = track
        self.exploration.set_state({"last_timestep" : 0})

    def _after_train(self):
        self.cache_data('train')

    def _before_evaluate(self, track):
        self.is_training = False
        self.current_track_name = track

    def _after_evaluate(self):
        self.cache_data('eval')
