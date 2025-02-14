import math
import numpy as np

import matplotlib
matplotlib.use('Agg') # NOTE: QT backend not working inside container
from matplotlib import pyplot as plt

from gazebo_sim.algorithm.DQN import DQN
from gazebo_sim.algorithm.QGMM import QGMM

from gazebo_sim.utils.buffer.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from gazebo_sim.utils.Evaluation import *
from gazebo_sim.utils.Caching import Cache


class LFLearner:
    def __init__(self, obs_space, config):
        self.config = config
        
        self.action_space = [i for i in range(config.output_shape[0])]
        self.batch_size = config.train_batch_size
        self.task = config.start_as_task

        self.epsilon = self.epsilon_init = config.initial_epsilon
        self.epsilon_delta = config.epsilon_delta
        self.eps_replay_factor = config.eps_replay_factor
        self.eps_min = config.final_epsilon
        self.last_dice = -1

        if config.algorithm == 'QGMM': self.algorithm = QGMM(obs_space, config)
        elif config.algorithm == 'DQN': self.algorithm = DQN(obs_space, config)
        else:
            raise Exception(f"No valid algorithm was set... use one of: ['QGMM', 'DQN'].")
        
        self.input_dims = obs_space
        self.buffer_type = config.replay_buffer
        self.buffer_size = config.capacity
        
        self.init_buffer(self.buffer_type, self.buffer_size, input_dims=self.input_dims)

        self.stats = {}

    # -----------------------------------> STATS

    def update_stats(self, key, value):
        try: self.stats[key].append(value)
        except: self.stats[key] = [value]
        
    def fetch_stats(self):
        return self.stats

    def cache_stats(self, mode): # NOTE: update cache whenever needed; serialize (via Exchanger) after each task
        for entry in self.stats:
            if entry.find(mode) == -1: 
                continue

            if entry not in Cache().object_registry:
                Cache().register_object(entry, None, category='data')
            
            current_object = Cache().object_registry[entry][0]
            try: 
                updated_object = np.stack([current_object, np.array(self.stats[entry])])  # NOTE: stack vs concatenate ?
            except: updated_object = np.array(self.stats[entry])
            Cache().update_object(entry, updated_object)

            self.stats[entry].clear()

    def gather_stats(self):
        for k,v in self.algorithm.algo_stats.items():
            self.update_stats(k, v)

    # -----------------------------------> PRE/POST ROUTINES

    def before_train(self):
        self.is_training = True
        self.task += 1

        if self.task == 1:
            self.epsilon = self.epsilon_init # reset epsilon to init
        if self.task > 1 or self.config.load_ckpt: 
            self.epsilon = self.epsilon_init * self.eps_replay_factor  # NOTE: scale initial_eps & eps_delta for replay training.        
            self.epsilon_delta = self.epsilon_delta * self.eps_replay_factor 

        self.algorithm.before_train(self.task)

    def after_train(self):
        self.algorithm.after_train(self.task)
        self.cache_stats('train')
        self.init_buffer(self.buffer_type, self.buffer_size, input_dims=self.input_dims) # clear buffer b4 training!

    def before_evaluate(self):
        self.is_training = False
        self.algorithm.before_evaluate(self.task)
        self.epsilon = 0.0
    
    def after_evaluate(self):
        self.algorithm.after_evaluate(self.task)
        self.cache_stats('eval')

    # -----------------------------------> BUFFER HANDLING

    def init_buffer(self, buffer_type, buffer_size, input_dims):
        if self.config.algorithm == 'QGMM':
            self.replay_buffer = ReplayBuffer(buffer_size, input_dims)
        else:
            if buffer_type == 'prioritized':
                self.replay_buffer = PrioritizedReplayBuffer(buffer_size, input_dims, self.config.per_alpha, self.config.per_beta, self.config.per_eps)
            else:
                self.replay_buffer = ReplayBuffer(buffer_size, input_dims)

    def store_transition(self, state, action, reward, new_state, done):
        """ stores a transition in replay buffer and optionally saves buffer content to file for offline pre-training. """
        self.replay_buffer.store_transition(state, action, reward, new_state, done)
        
        # NOTE: Code snippet to serialize the buffer's obs. data for pre-training of GMMs!
        # if self.replay_buffer.counter % self.config.capacity == 0 and (self.replay_buffer.counter > 1):
        #     nr = len(self.replay_buffer.state_memory)
        #     rb = self.replay_buffer
        #     target_q = (
        #         np.eye(len(self.action_space))[rb.action_memory]
        #         * rb.reward_memory[:, np.newaxis]
        #     )
        #     path = "/home/ak/lf-straight-5000.npz"
        #     print(f"saving .npz file to: {path}")
        #     np.savez(
        #         path,
        #         self.replay_buffer.state_memory,
        #         self.replay_buffer.state_memory,
        #         target_q,
        #         target_q,
        #     )
        
        # NOTE visualize sample buffer states (o_{t}, o_{t+1})
        # -----------------------------------
        #         
        # states, actions, rewards, states_, terminal = self.replay_buffer.sample_buffer(batch_size=64)
        # s1 = np.squeeze(states, axis=-1)
        # f, axes = plt.subplots(int(math.sqrt(s1.shape[0])), int(math.sqrt(s1.shape[0])))
        # for i, ax in enumerate(axes.ravel()):
        #     ax.imshow(s1[i])
        #     ax.set_axis_off()
        #     ax.set_xticklabels([])
        #     ax.set_yticklabels([])
        #     ax.set_aspect('equal')
        # plt.subplots_adjust(wspace=0.1, hspace=0.5)
        # plt.axis('off')
        # plt.savefig(f'{self.log_path}/states-{self.train_ctr}', bbox_inches='tight')
        # plt.close('all')
        
        # s2 = np.squeeze(states_, axis=-1)
        # f, axes = plt.subplots(int(math.sqrt(states_.shape[0])), int(math.sqrt(states_.shape[0])))
        # for i, ax in enumerate(axes.ravel()):
        #     ax.imshow(s2[i])
        #     ax.set_axis_off()
        #     ax.set_xticklabels([])
        #     ax.set_yticklabels([])
        #     ax.set_aspect('equal')
        # plt.subplots_adjust(wspace=0.1, hspace=0.5)
        # plt.axis('off')
        # plt.savefig(f'{self.log_path}/states_-{self.train_ctr}', bbox_inches='tight')
        # plt.close('all')

    # -----------------------------------> RL POLICY

    def learn(self):
        if not self.is_training:
            return  # skip if evaluating

        if self.replay_buffer.counter < self.batch_size:
            return  # if buffer is not full yet -> pass
        
        # FIXME: ugh, but2lazy
        if self.config.algorithm == 'QGMM':
            if self.algorithm.gmm_training_active and self.epsilon == self.eps_min:
                self.algorithm.gmm_training_active = False  # NOTE: disable GMM layer training when eps_min is reached

        states, actions, rewards, states_, terminal, batch_indices, importance = self.replay_buffer.sample_buffer(self.batch_size)

        self.algorithm.learn(self.task, states, actions, rewards, states_, terminal)
        
        if self.config.algorithm == 'DQN' and self.buffer_type == 'prioritized':
            self.replay_buffer.update_priorities(batch_indices, self.algorithm.td_error)
            self.algorithm.algo_stats.update({
                'per_importance': importance
            })
            
        self.algorithm.algo_stats.update({
            'epsilon': self.epsilon
        })

        self.update_epsilon()

        self.gather_stats()

    def choose_action(self, observation, force=-1):
        if force == -1:
            self.last_dice = np.random.random()
            if self.last_dice < self.epsilon:
                randomly_chosen = True
                actions = np.random.normal(size=(1, len(self.action_space)))
                action = int(np.argmax(actions))
            else:
                randomly_chosen = False
                state = observation
                actions = self.algorithm.model(state)
                action = int(np.argmax(actions, axis=1))
        else:
            randomly_chosen = False
            action = force
        
        return action, randomly_chosen

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_delta if self.epsilon > self.eps_min else self.eps_min
