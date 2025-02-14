import os
import time
import numpy as np
import tensorflow as tf

from gazebo_sim.algorithm.Algorithm import Algorithm
from gazebo_sim.model.DQN import build_dqn_models, build_dueling_models
from gazebo_sim.utils.Exchanging import Exchanger


class DQN(Algorithm):
    def __init__(
        self, obs_space, config
    ) -> None:
        self.config = config

        self.obs_space = obs_space
        self.n_actions = self.config.output_shape[0]
        self.train_batch_size = self.config.train_batch_size

        self.target_network_update_freq = self.config.dqn_target_network_update_freq

        self.gamma = self.config.gamma

        self.build_model()
        
        self.train_step = 0
        
    def build_model(self):
        if self.config.dqn_dueling:
            self.model, self.target_model = build_dueling_models(
                self.n_actions,
                self.obs_space,
                self.train_batch_size,
                self.config.dqn_fc1_dims,
                self.config.dqn_fc2_dims,
                self.config.dqn_adam_lr,
                self.config.dqn_target_network,
                self.config.load_ckpt
            )
        else:
            self.model, self.target_model = build_dqn_models(
                self.n_actions,
                self.obs_space,
                self.config.dqn_fc1_dims,
                self.config.dqn_fc2_dims,
                self.config.dqn_adam_lr,
                self.config.dqn_target_network,
                self.config.load_ckpt
            )

    def learn(self, task, states, actions, rewards, states_, dones):
        t_start = time.time_ns()

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)

        if self.config.dqn_target_network or self.config.dqn_dueling:
            self.learn_doubleq(states, actions, rewards, states_, dones)
        else:
            self.learn_vanilla(states, actions, rewards, states_, dones)

        t_end = time.time_ns()
        self.train_step_duration = np.divide(np.abs(np.diff([t_start, t_end])), 1e9)
        self.train_step += 1

        self.gather_stats(task)

    def learn_vanilla(self, states, actions, rewards, states_, dones):
        dqn_variables = self.model.trainable_variables
        with tf.GradientTape() as g:
            # Get Q values for current obs s_{t} using online model: Q(s, a, theta_i)
            online_q_cur = self.model(states)
            pred_q_values = tf.gather(online_q_cur, actions, batch_dims=1)
            # ------------------------------------------------------------------------>
            self.target_q_next_qs = tf.stop_gradient(self.model(states_))
            # bellman equation
            self.target_q_values = self.encode(self.target_q_next_qs, rewards, dones)
             # ------------------------------------------------------------------------>
            self.td_error = tf.abs(pred_q_values - self.target_q_values)
            loss = tf.reduce_mean(tf.square(self.td_error))
            gradients = g.gradient(loss, dqn_variables)
            self.model.optimizer.apply_gradients(zip(gradients, dqn_variables))
        del g

    def learn_doubleq(self, states, actions, rewards, states_, dones):
        # Q(s,a;θ) = r + γQ(s', argmax_{a'}Q(s',a';θ);θ')
        # θ: online net; θ': frozen (target) network
        # θ decides best next action a'; θ' evaluates action (Q-value estimation)
        dqn_variables = self.model.trainable_variables
        with tf.GradientTape() as g:
            # Get Q values for current obs s_{t} using online model: Q(s, a, theta_i)
            online_q_cur = self.model(states)
            pred_q_values = tf.gather(online_q_cur, actions, batch_dims=1)
            # ------------------------------------------------------------------------>
            # Get Q values for best actions in next state s_{t+1} using online model: max(Q(s', a', theta_i)) w.r.t a'
            online_q_next = tf.stop_gradient(self.model(states_))
            online_q_next_max = tf.argmax(online_q_next, axis=-1)
            online_q_next_action_mask = tf.one_hot(online_q_next_max, self.n_actions) # one hot mask for actions
            # ------------------------------------------------------------------------>
            target_q_next = tf.stop_gradient(self.target_model(states_))            
            # Get Q values from target network for next state s_{t+1} and chosen action
            self.target_q_next_qs = tf.reduce_sum(online_q_next_action_mask * target_q_next, axis=-1)
            # bellman equation
            self.target_q_values = rewards + self.gamma * self.target_q_next_qs * dones
             # ------------------------------------------------------------------------>
            self.td_error = tf.abs(pred_q_values - self.target_q_values)
            loss = tf.reduce_mean(tf.square(self.td_error))
            gradients = g.gradient(loss, dqn_variables)
            self.model.optimizer.apply_gradients(zip(gradients, dqn_variables))   
        del g

        self.update_model()

    def encode(self, q_next, rewards, dones):
        """ bellman equation """
        max_next_q_values = tf.reduce_max(q_next, axis=1)
        target_q_values = rewards + self.gamma * max_next_q_values * dones
        return target_q_values

    def update_model(self, force=False):
        if self.train_step == 0 and not force: return
        if (self.train_step % self.target_network_update_freq == 0) or force:
            self.copy_model_weights(self.model, self.target_model)

    def gather_stats(self, task):
        self.algo_stats = {
            'train_iter_time': self.train_step_duration,                    # time for current train step
            'train_q_next_qs': self.target_q_next_qs.numpy(),               # (N, num_actions)
            'train_target_qs': self.target_q_values.numpy(),                # (N, )
        }
        
    def after_train(self, task):
        if self.config.dqn_target_network or self.config.dqn_dueling:
            self.update_model(force=True)  # force update when training is done!
            
        if self.config.checkpointing:
            ckpt_path = Exchanger().path_register['checkpoint']
            ckpt_path_ = os.path.join(ckpt_path, f'{self.model.name.lower()}-T{task}')
            self.model.save_weights(ckpt_path_)