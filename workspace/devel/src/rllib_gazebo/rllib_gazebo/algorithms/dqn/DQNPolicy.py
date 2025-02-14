import logging
import numpy as np
import tree

from gymnasium.spaces import Space
from typing import Dict, List, Union

from ray.rllib.algorithms.simple_q.utils import Q_SCOPE, Q_TARGET_SCOPE

from ray.rllib.evaluation.postprocessing import adjust_nstep

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2

from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.eager_tf_policy import _disallow_var_creation
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import LearningRateSchedule

from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.annotations import (
    override,
    is_overridden,
)
from ray.rllib.utils.exploration import ParameterNoise
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.tf_utils import (
    huber_loss,
    l2_loss,
    make_tf_callable,
    minimize_and_clip,
    reduce_mean_ignore_inf,
)
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    ModelGradients,
    TensorType,
)

from ray.util.debug import log_once

from rllib_gazebo.algorithms.dqn.QTFModel import QTFModel

from rllib_gazebo.utils.Evaluation import *
from rllib_gazebo.utils.Caching import Cache

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)

# Importance sampling weights for prioritized replay
PRIO_WEIGHTS = "weights"


class QLoss:
    def __init__(
        self,
        q_t_selected: TensorType,
        q_logits_t_selected: TensorType,    # unused (dist DQN)
        q_tp1_best: TensorType,
        q_dist_tp1_best: TensorType,        # unused (dist DQN)
        importance_weights: TensorType,
        rewards: TensorType,
        done_mask: TensorType,
        gamma: float = 0.99,
        n_step: int = 1,
        num_atoms: int = 1,
        v_min: float = -10.0,   # unused (dist DQN)
        v_max: float = 10.0,    # unused (dist DQN)
        loss_fn=huber_loss,
    ):

        if num_atoms > 1: # INFO: DISTRIBUTED (MULTIAGENT) DQN IS DISABLED
            raise NotImplementedError('Distributional DQN is disabled.')
        else:
            q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best

            # compute RHS of bellman equation
            q_t_selected_target = rewards + gamma**n_step * q_tp1_best_masked

            # compute the error (potentially clipped)
            self.td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)

            self.loss = tf.reduce_mean(
                tf.cast(importance_weights, tf.float32) * loss_fn(self.td_error)
            )
            self.stats = {
                "mean_q": tf.reduce_mean(q_t_selected),
                "mean_td_error": tf.reduce_mean(self.td_error),
                "loss": self.loss,
            }

            # ----------  DEBUG PRINT
            #print(f'\033[93m[INFO]\033[0m [DQNPolicy]:',
            #    f'Q_T_SELECTED:\n{q_t_selected}\n'
            #    f'Q_TP1_BEST:\n{q_tp1_best}\n'
            #    f'Q_T_SELECTED_TARGET:\n{q_t_selected_target}\n'
            #    f'Q_LOSS: {self.loss}\n'
            #    f'TD_ERROR: {self.td_error}\n'
            #)

class DQNPolicy(EagerTFPolicyV2):
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            config: AlgorithmConfigDict,
            **kwargs,
        ):
        super().__init__(observation_space, action_space, config, **kwargs)

        self.enable_eager_execution_if_necessary() # RLLib stuff

        # store custom data
        self.data_dict = {}
        self.is_training = None
        self.current_track_name = None

        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])

        # Note: this is a bit ugly, but loss and optimizer initialization must
        # happen after all the MixIns are initialized.
        self.maybe_initialize_optimizer_and_loss()

        if config["target_network"] == True:
            model_vars = self.model.trainable_variables()
            target_model_vars = self.target_model.trainable_variables()

            @make_tf_callable(self.get_session()) # TargetNetworkMixin
            def update_target_fn(tau):
                tau = tf.convert_to_tensor(tau, dtype=tf.float32)
                update_target_expr = []
                assert len(model_vars) == len(target_model_vars), (
                    model_vars,
                    target_model_vars,
                )
                for var, var_target in zip(model_vars, target_model_vars):
                    update_target_expr.append(
                        var_target.assign(tau * var + (1.0 - tau) * var_target)
                    )
                    logger.debug("Update target op {}".format(var_target))
                return tf.group(*update_target_expr)

            # Hard initial update.
            self._do_update = update_target_fn
            self.update_target(tau=self.config.get("tau", 1.0))

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

    @property
    def q_func_vars(self):  # TargetNetworkMixin
        if not hasattr(self, "_q_func_vars"):
            self._q_func_vars = self.model.variables()
        return self._q_func_vars

    @property
    def target_q_func_vars(self):  # TargetNetworkMixin
        if not hasattr(self, "_target_q_func_vars"):
            self._target_q_func_vars = self.target_model.variables()
        return self._target_q_func_vars

    def update_target(self, tau: int = None) -> None:  # TargetNetworkMixin
        self._do_update(np.float32(tau or self.config.get("tau", 1.0)))

    def variables(self) -> List[TensorType]:  # TargetNetworkMixin
        return self.model.variables()

    # FIXME: This was broken with checkpointing, see if last line concering tau is even necessary...
    # def set_weights(self, weights):  # TargetNetworkMixin
    #     self.set_weights(self, weights)
    #     self.update_target(self.config.get("tau", 1.0))

    def compute_td_error(
        self, obs_t, act_t, rew_t, obs_tp1, terminateds_mask, importance_weights
    ):
        ''' This allows us to prioritize on the worker side. '''
        # Do forward pass on loss to update td error attribute
        self.loss(
            self,
            self.model,
            None,
            {
                SampleBatch.CUR_OBS: tf.convert_to_tensor(obs_t),
                SampleBatch.ACTIONS: tf.convert_to_tensor(act_t),
                SampleBatch.REWARDS: tf.convert_to_tensor(rew_t),
                SampleBatch.NEXT_OBS: tf.convert_to_tensor(obs_tp1),
                SampleBatch.TERMINATEDS: tf.convert_to_tensor(terminateds_mask),
                PRIO_WEIGHTS: tf.convert_to_tensor(importance_weights),
            },
        )
        return self.q_loss.td_error

    @override(EagerTFPolicyV2)
    def make_model(self) -> ModelV2:

        # if self.config["hiddens"]:
        #     # try to infer the last layer size, otherwise fall back to 256
        #     num_outputs = ([256] + list(self.config["model"]["fcnet_hiddens"]))[-1]
        #     self.config["model"]["no_final_linear"] = True
        # else:
        num_outputs = self.action_space.n

        self.model = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=num_outputs,
            model_config=self.config["model"],
            framework="tf",
            model_interface=QTFModel,
            name=Q_SCOPE,
            num_atoms=self.config["num_atoms"],
            dueling=self.config["dueling"],
            q_hiddens=self.config["hiddens"],
            use_noisy=self.config["noisy"],
            v_min=self.config["v_min"],
            v_max=self.config["v_max"],
            sigma0=self.config["sigma0"],
            add_layer_norm=isinstance(getattr(self, "exploration", None), ParameterNoise)
            or self.config["exploration_config"]["type"] == "ParameterNoise",
        )

        if self.config["target_network"] == True:
            self.target_model = ModelCatalog.get_model_v2(
                obs_space=self.observation_space,
                action_space=self.action_space,
                num_outputs=num_outputs,
                model_config=self.config["model"],
                framework="tf",
                model_interface=QTFModel,
                name=Q_TARGET_SCOPE,
                num_atoms=self.config["num_atoms"],
                dueling=self.config["dueling"],
                q_hiddens=self.config["hiddens"],
                use_noisy=self.config["noisy"],
                v_min=self.config["v_min"],
                v_max=self.config["v_max"],
                sigma0=self.config["sigma0"],
                add_layer_norm=isinstance(getattr(self, "exploration", None), ParameterNoise)
                or self.config["exploration_config"]["type"] == "ParameterNoise",
            )

        return self.model

    @override(EagerTFPolicyV2)
    def loss(self, model, _, train_batch: SampleBatch) -> TensorType:
        """Constructs the loss for DQNTFPolicy.

        Args:
            policy: The Policy to calculate the loss for.
            model (ModelV2): The Model to calculate the loss for.
            train_batch: The training data.

        Returns:
            TensorType: A single loss tensor.
        """
        config = self.config

        if config["target_network"] == True:
            target_model = self.target_model
        else:
            target_model = model

        # ---------- FORWARD PASS: q network eval on s_t
        self.q_t, q_logits_t, q_dist_t, _ = self.compute_q_values(
            model,
            SampleBatch({"obs": train_batch[SampleBatch.CUR_OBS]}),
            state_batches=None,
            explore=False,
        )

        # ---------- FORWARD PASS: q network eval on s_(t+1)
        q_tp1, q_logits_tp1, q_dist_tp1, _ = self.compute_q_values(
            target_model,
            SampleBatch({"obs": train_batch[SampleBatch.NEXT_OBS]}),
            state_batches=None,
            explore=False,
        )

        if config["target_network"] == True: # update target model variables
            if not hasattr(self, "target_q_func_vars"):
                self.target_q_func_vars = self.target_model.variables()

        # ---------- Q SCORE SELECTION/MASKING
        # q scores for actions which we know were selected in the given state.
        one_hot_selection = tf.one_hot(
            tf.cast(train_batch[SampleBatch.ACTIONS], tf.int32), self.action_space.n
        )
        q_t_selected = tf.reduce_sum(self.q_t * one_hot_selection, 1)
        q_logits_t_selected = tf.reduce_sum(
            q_logits_t * tf.expand_dims(one_hot_selection, -1), 1
        )

        # ----------  ESTIMATE COMPUTATION FROM TARGET MODEL
        if config["double_q"]:
            (
                q_tp1_using_online_net,
                q_logits_tp1_using_online_net,
                q_dist_tp1_using_online_net,
                _,
            ) = self.compute_q_values(
                model,
                SampleBatch({"obs": train_batch[SampleBatch.NEXT_OBS]}),
                state_batches=None,
                explore=False,
            )
            q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
            q_tp1_best_one_hot_selection = tf.one_hot(
                q_tp1_best_using_online_net, self.action_space.n
            )
            q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
            q_dist_tp1_best = tf.reduce_sum(
                q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1
            )
        else:
            q_tp1_best_one_hot_selection = tf.one_hot(
                tf.argmax(q_tp1, 1), self.action_space.n
            )
            q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
            q_dist_tp1_best = tf.reduce_sum(
                q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1
            )

        loss_fn = huber_loss if self.config["td_error_loss_fn"] == "huber" else l2_loss

        self.q_loss = QLoss(
            q_t_selected,
            q_logits_t_selected,
            q_tp1_best,
            q_dist_tp1_best,
            train_batch[PRIO_WEIGHTS],
            tf.cast(train_batch[SampleBatch.REWARDS], tf.float32),
            tf.cast(train_batch[SampleBatch.TERMINATEDS], tf.float32),
            config["gamma"],
            config["n_step"],
            config["num_atoms"],
            config["v_min"],
            config["v_max"],
            loss_fn,
        )

        return self.q_loss.loss

    @override(EagerTFPolicyV2)
    def compute_gradients_fn(
        self, optimizer: "tf.keras.optimizers.Optimizer", loss: TensorType
    ) -> ModelGradients:
        if not hasattr(self, "q_func_vars"):
            self.q_func_vars = self.model.variables()

        return minimize_and_clip(
            optimizer,
            loss,
            var_list=self.q_func_vars,
            clip_val=self.config["grad_clip"],
        )

    @override(EagerTFPolicyV2)
    def extra_learn_fetches_fn(self) -> Dict[str, TensorType]:
        return {}

    @override(EagerTFPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        """Returns the learning rate in a stats dict.

        Args:
            policy: The Policy object.
            train_batch: The data used for training.

        Returns:
            Dict[str, TensorType]: The stats dict.
        """

        return dict(
            {"cur_lr": tf.cast(self.cur_lr, tf.float64),},
            **self.q_loss.stats,
        )

    def compute_q_values(
        policy: Policy,
        model: ModelV2,
        input_batch: SampleBatch,
        state_batches=None,
        seq_lens=None,
        explore=None,
        is_training: bool = False,
    ):
        config = policy.config

        model_out, state = model(input_batch, state_batches or [], seq_lens)

        if config["num_atoms"] > 1: # INFO: DISTRIBUTED (MULTIAGENT) DQN IS DISABLED
            raise NotImplementedError('Distributional DQN is disabled.')
        else:
            (action_scores, logits, dist) = model.get_q_value_distributions(model_out)

        if config["dueling"]:
            state_score = model.get_state_value(model_out)
            if config["num_atoms"] > 1:
                raise NotImplementedError('Distributional DQN is disabled.')
            else:
                action_scores_mean = reduce_mean_ignore_inf(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(
                    action_scores_mean, 1
                )
                value = state_score + action_scores_centered
        else:
            value = action_scores

        return value, logits, dist, state

    @override(EagerTFPolicyV2)
    def optimizer(
        self,
    ) -> Union["tf.keras.optimizers.Optimizer", List["tf.keras.optimizers.Optimizer"]]:
        """TF optimizer to use for policy optimization.

        Returns:
            A local optimizer or a list of local optimizers to use for this
                Policy's Model.
        """
        if self.config["framework"] == "tf2":
            return tf.keras.optimizers.Adam(
                learning_rate=self.cur_lr,
                beta_1=self.config["adam_beta1"],
                beta_2=self.config["adam_beta2"],
            )
        else:
            raise NotImplementedError('Non tf2 mode is not supported!')

    @override(EagerTFPolicyV2)
    def postprocess_trajectory(
        policy: Policy, batch: SampleBatch, other_agent=None, episode=None
    ) -> SampleBatch:
        # N-step Q adjustments.
        if policy.config["n_step"] > 1:
            adjust_nstep(policy.config["n_step"], policy.config["gamma"], batch)

        # Create dummy prio-weights (1.0) in case we don't have any in
        # the batch.
        if PRIO_WEIGHTS not in batch:
            batch[PRIO_WEIGHTS] = np.ones_like(batch[SampleBatch.REWARDS])

        # Prioritize on the worker side.
        if batch.count > 0 and policy.config["replay_buffer_config"].get(
            "worker_side_prioritization", False
        ):
            td_errors = policy.compute_td_error(
                batch[SampleBatch.OBS],
                batch[SampleBatch.ACTIONS],
                batch[SampleBatch.REWARDS],
                batch[SampleBatch.NEXT_OBS],
                batch[SampleBatch.TERMINATEDS],
                batch[PRIO_WEIGHTS],
            )
            # Retain compatibility with old-style Replay args
            epsilon = policy.config.get("replay_buffer_config", {}).get(
                "prioritized_replay_eps"
            ) or policy.config.get("prioritized_replay_eps")
            if epsilon is None:
                raise ValueError("prioritized_replay_eps not defined in config.")

            new_priorities = np.abs(convert_to_numpy(td_errors)) + epsilon
            batch[PRIO_WEIGHTS] = new_priorities

        return batch

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
            #FIXME: for now we use append and clear the lists after each track.
            # this shouldn't be too critical regarding run-time -> append has amortized O(1).
            # see if there's anoother option since we can not pre-allocate space when using unit = episodes.
            if self.is_training == True: self.update_data('train_online_q_values', dist_inputs)
            if self.is_training == False: self.update_data('eval_online_q_values', dist_inputs)

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
