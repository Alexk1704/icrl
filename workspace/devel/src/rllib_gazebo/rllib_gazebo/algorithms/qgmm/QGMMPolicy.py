import os
import logging
import numpy as np
import tensorflow as tf
import tree

from gymnasium.spaces import Space
from typing import *

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution

from ray.rllib.policy.eager_tf_policy import _disallow_var_creation
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.annotations import (
    override,
    is_overridden,
)
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    LocalOptimizer,
    ModelGradients,
    TensorType,
)

from rllib_gazebo.utils.Evaluation import *
from rllib_gazebo.utils.Caching import Cache

from cl_replay.architecture.ar.generator import DCGMM_Generator

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)

# Importance sampling weights for prioritized replay
PRIO_WEIGHTS = "weights"


class QGMMPolicy(EagerTFPolicyV2):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        config: AlgorithmConfigDict,
        **kwargs,
        ):
        super().__init__(observation_space, action_space, config)

        self.enable_eager_execution_if_necessary() # RLLib stuff

        # store custom data
        self.data_dict = {}
        self.is_training = None
        self.current_track_name = None
        self.replay_active = None

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
    def make_model(self) -> ModelV2:
        # -------------------------------- Q-MODEL -> Short-Term-Memory / Learner (accessible via policy.model.base_model)
        self.STM = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=self.action_space.n,
            model_config=self.config['model'],
            framework=self.config['framework'],
            name='STM', stm=True,
        )

        # -------------------------------- Q-MODEL -> Long-Term-Memory / Generator
        self.LTM = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=self.action_space.n,
            model_config=self.config['model'],
            framework=self.config['framework'],
            name='LTM', stm=False,
        )

        if len(self.LTM.base_model.layers) == len(self.STM.base_model.layers):
            self.generator_top_included = True
            self.sampling_layer = -1
        else:
            self.generator_top_included = False
            self.sampling_layer = -2

        # -------------------------------- GENERATOR FUNC. -> Use LTM to sample observations
        data_dims = list(self.observation_space.shape)
        data_dims.append(self.action_space.n)
        self.generator = DCGMM_Generator(
            model = self.LTM.base_model,
            data_dims = data_dims,
        )

        self.opt_and_layers = self.STM.base_model.opt_and_layers

        self.var_losses = [np.nan for _ in self.STM.base_model.opt_and_layers]
        self.q_losses   = [np.nan for _ in self.STM.base_model.opt_and_layers]
        self.full_q     = [np.nan] * self.action_space.n

        return self.STM

    def _compute_q_values(self, model:ModelV2, obs_batch: TensorType, is_training=None) -> TensorType:
        _is_training = (
            is_training
            if is_training is not None
            else self._get_is_training_placeholder()
        )

        model_out, _ = model(SampleBatch(obs=obs_batch, _is_training=_is_training), [], None)

        return model_out

    @override(EagerTFPolicyV2)
    def extra_learn_fetches_fn(self) -> Dict[str, TensorType]:
        """Extra stats to be reported after gradient computation. """
        return {
            "q_losses": self.q_losses,
            "var_losses": self.var_losses,
            "full_q": self.q_next,
        }

    @override(EagerTFPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        """Stats function. Returns a dict of statistics.

        Args:
            train_batch: The SampleBatch (already) used for training.

        Returns:
            The stats dict.
        """
        return {}

    # --------------------------------------------> CUSTOM START (AK)

    """
    def learn_on_batch(self, postprocessed_batch):
        ''' We could overwrite this aswell, however, since this fn simply calls the helper, only _learn_on_batch_helper() is modified. '''
        return super(CustomQTF2Policy, self).learn_on_batch(postprocessed_batch)
    """

    def _learn_on_batch_helper(self, samples, _ray_trace_ctx=None):
        ''' this fn is overwritten and gets called by self.learn_on_batch() to perform a single training iteration. '''
        self._re_trace_counter += 1                         # RLlib; increase eager retracing count

        # ---------- LTM: VARIANT GENERATION
        if self.replay_active == True:                      # only perform generation if allowed to, i.e., we are past initial training phase
            variants, _ = self.generate_variants(           # use current state of the generator for variant generation based on current obs. batch s_(t)
                samples, generate_labels=False)
            variants_xs = tf.constant(variants)             # convert variants from numPy to EagerTensor
            variants_ys = self.LTM.base_model(variants)     # if LTM includes the top (readout) layer; also generate pseudo labels/q-values
            # ---------- TRAINING/OPT.
            self.model.base_model.pre_train_step()

            with tf.GradientTape(persistent=True) as q_var_tape:
                self.var_losses = self.var_loss(self.model, variants_xs, variants_ys)
            # ---- KERAS METRICS
            self.update_keras_metrics(self.opt_and_layers, self.var_losses)
            # ---- OPTIMIZATION
            grads_vars = self.compute_gradients_fn(self.optimizer, self.var_losses, q_var_tape)
            grads = self.opt_keras_model(self.opt_and_layers, grads_vars)

            del q_var_tape

            self.model.base_model.post_train_step()
            # ----------
        # ----------

        # ---------- STM: Q-LEARNING
        self.model.base_model.pre_train_step()

        with tf.GradientTape(persistent=True) as q_tape:
            self.q_losses = self.loss(self.model, self.dist_class, samples)
        # ---- KERAS METRICS
        self.update_keras_metrics(self.opt_and_layers, self.q_losses)
        # ---- OPTIMIZATION
        grads_vars = self.compute_gradients_fn(self.optimizer, self.q_losses, q_tape)
        grads = self.opt_keras_model(self.opt_and_layers, grads_vars)

        del q_tape

        self.model.base_model.post_train_step()
        # ----------

        stats = self._stats(samples, grads) # RLLib, logging
        return stats

    def update_keras_metrics(self, opt_and_layers, loss):
        ''' set layer metrics '''
        for (_, _layer), _loss in zip(opt_and_layers, loss):
            m = _layer.get_layer_metrics()
            if m is not None: m[0].update_state(_loss)

    def opt_keras_model(self, opt_and_layers, grads_vars):
        ''' optimization / gradient application '''
        grads = []
        for _grads_vars, (_opt, _) in zip(grads_vars, opt_and_layers):
            grads += [g for g, _ in _grads_vars]
            _opt.apply_gradients(_grads_vars)
        return grads

    def get_keras_loss(self, opt_and_layers):
        ''' loss calculation '''
        losses = []
        for i, (_, _layer) in enumerate(opt_and_layers):
            raw_loss = _layer.loss_fn(y_true=None, y_pred=_layer.get_fwd_result())
            _layer.set_layer_loss(tf.reduce_mean(raw_loss) * -1.)
            _loss = _layer.get_layer_loss()
            losses += [_loss]
        return losses

    def var_loss(self, model: Union[ModelV2, "tf.keras.Model"], variants_xs: Union[TensorType, None], variants_ys: Union[TensorType, None]):
        ''' performs the loss computation for generated variants '''
        var_losses = []
        # ----- LTM
        var_cur = self.model.base_model(variants_xs)                # STM forward call on generated data, returns current q value estimation for variant data

        # ----- UPDATE GMM
        _var_losses = self.get_keras_loss(self.opt_and_layers[:-1]) # get losses, except for top readout (in case of )
        var_losses += _var_losses

        if self.generator_top_included == True:
            # ----- UPDATE READOUT
            _var_ro_loss = self.opt_and_layers[-1][1].loss_fn(
                y_true=tf.stop_gradient(variants_ys),               # use "fake" variant labels instead of bellman-equation output
                y_pred=var_cur)                                     # current q value estimation for variant
            var_losses.append(_var_ro_loss)
        # -----

        # ----------  DEBUG PRINT
        #print(f'\033[93m[INFO]\033[0m [QTFPolicy]: VARIANT LOSSES: {var_losses}')
        return var_losses

    @override(EagerTFPolicyV2)
    def optimizer(self):
        return None

    @override(EagerTFPolicyV2)
    def loss(self, model: Union[ModelV2, "tf.keras.Model"], dist_class: Type[TFActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        ''' performs the Q-loss computation using the bellman equation '''
        # ---------- DATA
        q_losses = []                                          # list holding loss tensor's
        s_cur = train_batch[SampleBatch.CUR_OBS]               # s_(t)      -> train input (current)
        s_next = train_batch[SampleBatch.NEXT_OBS]             # s_(t+1)    -> used for bellman equation (next)

        # ---------- FORWARD PASS: q network eval on s_t
        self.q_cur = self._compute_q_values(model, s_cur)      # Feed s_t into the network to get Q(s,a;θ)/y_pred/logits

        # ----- STM
        # ---------- GMM LOSS: accumulate losses after FWD on s_t (top readout is excluded)
        _losses = self.get_keras_loss(opt_and_layers=self.opt_and_layers[:-1])
        q_losses += _losses

        # ---------- FORWARD PASS: q network eval on s_(t+1)
        self.q_next = self._compute_q_values(model, s_next)  # Feed s_t+1 into the network to get Q(s',a';θ)/y_pred/logits

        # ---------- Q SCORE SELECTION/MASKING
        one_hot_selection = tf.one_hot(
            tf.cast(train_batch[SampleBatch.ACTIONS], tf.int32), self.action_space.n
        )
        q_cur_selected = tf.reduce_sum(self.q_cur * one_hot_selection, 1)    # Chose max action 'max_a' for Q(s,a;θ), get the q value for selected action

        # ----------  ESTIMATE COMPUTATION FROM TARGET MODEL
        dones = tf.cast(train_batch[SampleBatch.TERMINATEDS], tf.float32)   # 'dones' signals if terminal state was reached for action
        q_next_onehot = tf.one_hot(
            tf.argmax(self.q_next, 1), self.action_space.n
        )
        q_next_best = tf.reduce_sum(self.q_next * q_next_onehot, 1)  # Chose max action 'max_a' for Q(s',a';θ), compute estimate of best possible value
        q_next_best_masked = (1.0 - dones) * q_next_best        # mask max action for t+1 to consider if terminal state was reached

        # ----------  BELLMAN EQUATION
        q_next_targets = (                                      # get targets for policy network: g * max_a + r_t
            tf.cast(train_batch[SampleBatch.REWARDS], tf.float32)
            + self.config["gamma"] * q_next_best_masked
        )

        # ---------- READOUT LOSS
        self.q_loss = self.opt_and_layers[-1][1].loss_fn(
            y_true=tf.stop_gradient(q_next_targets),    # output of bellman equation (q_next_targets) -> labels/y_truen (based on obs. t+1)
            y_pred=q_cur_selected)                      # q value selection (q_cur_selected)          -> logits/y_pred  (based on current obs. t)
        q_losses += [self.q_loss]

        # ----------  DEBUG PRINT
        #print(f'\033[93m[INFO]\033[0m [QTFPolicy]:',
        #    f'\nQ_CURRENT:\n{self.q_cur}\n'
        #    f'Q_NEXT:\n{self.q_next}\n'
        #    f'SELCTED_ACTION:\n{one_hot_selection}\n'
        #    f'Q_CURRENT_SELECTED:\n{q_cur_selected}\n'
        #    f'Q_NEXT_ONEHOT:\n{q_next_onehot}\n'
        #    f'Q_NEXT_BEST:\n{q_next_best}\n'
        #    f'Q_NEXT_BEST_MASKED:\n{q_next_best_masked}\n'
        #    f'Q_NEXT_TARGETS:\n{q_next_targets}\n'
        #    f'Q_LOSSES: {q_losses}\n'
        #    f'Q_LOSS: {self.q_loss}\n'
        #)
        return q_losses

    @override(EagerTFPolicyV2)
    def compute_gradients_fn(self, optimizer: LocalOptimizer, loss: TensorType, tape: "tf.keras.GradientTape") -> ModelGradients:
        '''
            --opt: rllib default, we can ignore this and use our own optimizer instances
            --loss: list of (q-,var-)losses returned by the loss functions

            Overwriting the default behaviour to utilize multiple optimizer instances.
            This performs a separate gradient computation for each (layer,loss) pair of the policy model.
        '''
        grads_vars = []
        for (_opt, _layer), _loss in zip(self.model.base_model.opt_and_layers, loss):
            _vars = _layer.trainable_variables
            _grads = tape.gradient(_loss, _vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            _grads_vars = self.model.base_model.factor_gradients(zip(_grads, _vars), _layer.get_grad_factors())
            grads_vars += [_grads_vars]
        return grads_vars

    @override(EagerTFPolicyV2)
    def apply_gradients_fn(self, optimizer, grads):
        return None

    def generate_variants(self, train_batch: SampleBatch, generate_labels=False):
        """ takes observations to train on (SampleBatch) & generates a numPy array of variants with same dimensions from the LTM """
        current_obs = train_batch[SampleBatch.CUR_OBS].numpy()
        gen_xs, gen_ys = self.generator.generate_data(
            xs=current_obs, generate_labels=generate_labels,
            stg=current_obs.shape[0], sbs=current_obs.shape[0],
            variants=True, sampling_layer=self.sampling_layer,
        )

        return gen_xs, gen_ys

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
        self.replay_active = True
        if self.current_track_name is None:
            self.replay_active = False

        self.is_training = True
        self.current_track_name = track
        self.exploration.set_state({"last_timestep" : 0})

        for t_c in self.model.train_callbacks: # train callback trigger
            t_c.on_train_begin()

        if self.replay_active == True: # copy model weights from current base_model to LTM
            self.copy_model_weights(source=self.model.base_model, target=self.LTM.base_model) # copy weights of base_model to LTM
            self.model.base_model.reset() # reset somSigma

    def _after_train(self):
        for t_c in self.model.train_callbacks: t_c.on_train_end()

        self.cache_data('train')

    def _before_evaluate(self, track):
        self.is_training = False
        self.current_track_name = track

        for e_c in self.model.eval_callbacks: e_c.on_test_begin()

    def _after_evaluate(self):
        for e_c in self.model.eval_callbacks: e_c.on_test_end()

        self.cache_data('eval')

    def copy_model_weights(self, source: Union[ModelV2, "tf.keras.Model"], target: Union[ModelV2, "tf.keras.Model"]):
        ''' in-memory copy of model weights '''
        if self.generator_top_included == False:
            source_layers = source.layers[:-1]
        else: # we assume that the only difference between the source & target models is the top-level readout
            source_layers = source.layers

        for source_layer, target_layer in zip(source_layers, target.layers):
            source_weights = source_layer.get_weights()
            target_layer.set_weights(source_weights)
            target_weights = target_layer.get_weights()

            if source_weights and all(tf.nest.map_structure(np.array_equal, source_weights, target_weights)):
                print(f'\033[93m[INFO]\033[0m [QTFPolicy]: WEIGHT TRANSFER: {source.name}-{source_layer.name} -> {target.name}-{target_layer.name}')

    def save_model_weights(self, model: Union[ModelV2, "tf.keras.Model"], chkpt_dir: str, save_str: str):
        ''' save a checkpoint file on fs '''
        chkpt_filename = os.path.join(chkpt_dir, f'{save_str}-{model.base_model.name.lower()}-clone.ckpt')
        model.base_model.save_weights(chkpt_filename)

        return chkpt_filename

    def load_model_weights(self, model: Union[ModelV2, "tf.keras.Model"], chkpt_path: str):
        ''' load from a checkpoint file on fs '''
        try:
            model.load_weights(chkpt_path)
            print(f'loading model state from checkpoint: "{chkpt_path}"...')
        except Exception as ex:
            import traceback
            print(traceback.format_exc())
            print(f'A problem was encountered loading model state from checkpoint: "{chkpt_path}": {ex}')
        raise ex

    @override(EagerTFPolicyV2)
    def get_weights(self, as_dict=False):
        return super().get_weights(as_dict)

    @override(EagerTFPolicyV2)
    def set_weights(self, weights):
        super().set_weights(weights)
