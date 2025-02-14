import os
import time
import itertools
import threading

import numpy as np
import gymnasium as gym

import rclpy
import rclpy.qos as QoS

from rclpy.node import Node

from ros_gz_interfaces.srv import *
from ros_gz_interfaces.msg import Clock

from custom_interfaces.srv import *

from std_msgs.msg import Empty as EmptyMsg

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from gazebo_sim.utils.ArgsConfig import Args
from gazebo_sim.utils.Caching import Cache
from gazebo_sim.utils.Evaluation import *
from gazebo_sim.utils.Logging import Logger
from gazebo_sim.utils.Tasks import Tracks


class ThreadSafeSingleton(type):
    _instances = {}
    _singleton_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # double-checked locking pattern (https://en.wikipedia.org/wiki/Double-checked_locking)
        if cls not in cls._instances:
            with cls._singleton_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(ThreadSafeSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class EnvironmentConfig():
    def __init__(self) -> None:
        for name, value in vars(Args().args).items():
            setattr(self, name, value)

        self.observation_dims = {k: v for k, v in zip('xyz', self.state_shape)}
        self.desired_line_center = np.divide(self.observation_dims['y'], 2)

        start, stop, number = self.state_quantization
        self.state_space = np.linspace(start, stop, int(number))

        start, stop, number = self.action_quantization
        self.action_space = itertools.product(
            np.linspace(start, stop, int(number)),
            np.linspace(start, stop, int(number)),
        )

        self.action_entries = []
        # NOTE: hard coded action space overwrites params!
        self.action_space = [
            [0.1, 0.3], [0.1, 0.5], [0.2, 0.4], [0.2, 0.3], 
            [0.3, 0.3], 
            [0.3, 0.2], [0.4, 0.2], [0.5, 0.1], [0.3, 0.1]
        ]
        
        for wheel_speeds in self.action_space:
            diff = np.diff(wheel_speeds)
            if diff == 0: self.action_entries.append(TwistAction('forward', wheel_speeds))
            elif diff < 0: self.action_entries.append(TwistAction('right', wheel_speeds))
            elif diff > 0: self.action_entries.append(TwistAction('left', wheel_speeds))

        
class EnvironmentData():
    def __init__(self, config:EnvironmentConfig) -> None:
        self.env_config = config

        self.context = 0
        self.episode = 0
        self.tick = 0
        self.without_line = 0

        self.stats = {
            'length': {'last': 0., 'curr': 0., 'acc': 0., 'avg': 0., 'min': 0., 'max': 0.},
            'return': {'last': 0., 'curr': 0., 'acc': 0., 'avg': 0., 'min': 0., 'max': 0.},
        }

        self.storage_location = os.path.join(self.env_config.root_dir, self.env_config.exp_id, 'debug', 'Raw')

        self.observations = np.zeros((self.env_config.max_steps_per_episode, np.prod(list(self.env_config.observation_dims.values()))), np.uint8)
        self.actions = np.zeros((self.env_config.max_steps_per_episode, 1), np.uint8)
        self.rewards = np.zeros((self.env_config.max_steps_per_episode, 1), np.float32)

        self.clocks = np.zeros((self.env_config.max_steps_per_episode, 2), np.float32)
        self.normalized_states = np.zeros((self.env_config.max_steps_per_episode, 1), np.float32)
        self.normalized_actions = np.zeros((self.env_config.max_steps_per_episode, 2), np.float32)
        self.normalized_rewards = np.zeros((self.env_config.max_steps_per_episode, 1), np.float32)

        self.sequence_placeholder = np.zeros((self.env_config.sequence_length, self.observations.shape[1]), np.uint8)


    def current_data(self, array):
        return array[:self.tick]

    def store_data(self, name, array):
        np.save(os.path.join(self.storage_location, name), self.current_data(array))

    def reinit_stats(self):
        self.stats = {
            'length': {'last': 0., 'curr': 0., 'acc': 0., 'avg': 0., 'min': 0., 'max': 0.},
            'return': {'last': 0., 'curr': 0., 'acc': 0., 'avg': 0., 'min': 0., 'max': 0.},
        }

    def update_stats(self):
        self.stats['length']['last'] = self.stats['length']['curr']
        self.stats['length']['curr'] = self.tick
        self.stats['length']['acc'] += self.tick
        self.stats['length']['avg'] = self.stats['length']['acc'] / self.episode
        if self.tick < self.stats['length']['min']: self.stats['length']['min'] = self.tick
        if self.tick > self.stats['length']['max']: self.stats['length']['max'] = self.tick

        temp = np.sum(self.rewards[:self.tick])
        self.stats['return']['last'] = self.stats['return']['curr']
        self.stats['return']['curr'] = temp
        self.stats['return']['acc'] += temp
        self.stats['return']['avg'] = self.stats['return']['acc'] / self.episode
        if temp < self.stats['return']['min']: self.stats['return']['min'] = temp
        if temp > self.stats['return']['max']: self.stats['return']['max'] = temp

    def switch(self):
        if self.env_config.debug: Logger().debug(f'switch invoked')

        if self.context > 0:
            self.update_stats()

            if self.env_config.debug:
                Logger().debug(f'stats={self.stats}')

        self.context += 1
        self.episode = 0

        self.reinit_stats()

    def reset(self):
        if self.env_config.debug: Logger().debug(f'reset invoked')

        if self.episode > 0:
            self.update_stats()

            if self.env_config.debug:
                Logger().debug(f'stats={self.stats}')

        self.episode += 1
        self.tick = 0

        self.without_line = 0

    def step(self):
        self.tick += 1

    def prepare_observation(self, tick):
        observation = self.observations[tick]
        rgb_image = observation.reshape(*self.env_config.observation_dims.values())
        grayscale_image = np.average(np.divide(rgb_image, 255., dtype=np.float32), axis=-1)

        return grayscale_image

    def prepare_action(self, tick):
        index = self.actions[tick]
        action = self.env_config.action_entries[index[0]]

        return action

    def normalize_state(self, tick):
        '''
        line offset (deviation)
        line angle (alignment)
        line slope (vertical course)
        line tilt (horizontal course)
        line curvature (skewness)

        direction, movement etc. only via sequence
        '''
        image = self.prepare_observation(tick)

        # set once in init the corresponding function, which to use
        if self.env_config.line_mode == 'c': state = self.quantize_line_position(image)
        elif self.env_config.line_mode == 'l': state = self.quantize_line_position(image, 'l')
        elif self.env_config.line_mode == 'r': state = self.quantize_line_position(image, 'r')

        if not np.isnan(state):
            return normalize(state, 0, self.env_config.observation_dims['y'], *self.env_config.state_normalization)
        return np.nan

    def normalize_action(self, tick):
        action = self.prepare_action(tick)
        normed_action = normalize(action.wheel_speeds, *self.env_config.action_quantization[:2], *self.env_config.action_normalization)

        return normed_action

    def quantize_line_position(self, image:np.ndarray, mode:str='c', simple:bool=True):
        if simple: image = np.mean(image, axis=0, keepdims=True)

        mask = (image < self.env_config.line_threshold)

        if np.all(~mask): return np.nan
        # if np.all(mask): return np.nan

        # l_white_index = 0 + np.mean(np.argmin(mask, axis=-1))
        l_black_index = 0 + np.mean(np.argmax(mask, axis=-1))

        mask = np.flip(mask, axis=-1)

        # r_white_index = self.env_config.observation_dims['y'] - np.mean(np.argmin(mask, axis=-1)) -1
        r_black_index = self.env_config.observation_dims['y'] - np.mean(np.argmax(mask, axis=-1)) -1

        if mode == 'c':
            # if l_black_index <= l_white_index or r_black_index >= r_white_index: return np.nan
            # else: return np.mean([l_black_index, r_black_index])
            return np.mean([l_black_index, r_black_index])
        elif mode == 'l':
            # if l_black_index <= l_white_index: return np.nan
            # else: return l_black_index
            return l_black_index
        elif mode == 'r':
            # if r_black_index >= r_white_index: return np.nan
            # else: return r_black_index
            return r_black_index

    def check_done(self, state):
        if np.isnan(state): return (True, True)  # in case no valid obs.
        if not self.tick < (self.env_config.max_steps_per_episode - 1): return (False, True)  # truncated
        if not self.without_line < self.env_config.max_steps_without_line: return (True, False)  # terminated
        return (False, False)  # all g00d

    def make_assessment(self, tick:int):
        # calculate deltas (state improvement, action deviation)
        current = np.concatenate([
            self.normalized_states[tick],
            np.average(self.normalized_actions[tick], keepdims=True),
        ])

        if tick == 1:
            previous = current
        else:
            # use here the values depending on the reward function components
            # action_diff-diff -> temporary store the previous action diff and compare with
            previous = np.concatenate([
                self.normalized_states[tick - 1],
                np.average(self.normalized_actions[tick - 1], keepdims=True),
            ])

        return np.diff([previous, current], axis=0).squeeze()

    def calculate_reward(self, tick:int) -> tuple:
        assert tick > 0, 'tick should never be 0!'
        # print("TICK: ", tick)
        # print("------ NORMED ACTIONS ARRAY:\n", self.normalized_actions[:10])
        # print("------ NORMED STATES ARRAY:\n", self.normalized_states[:10])
        state = self.normalized_states[tick]
        action = np.average(self.normalized_actions[tick-1], keepdims=True)
        if tick == 1:
            prev_state = state
            prev_action = action
        else:
            prev_state = self.normalized_states[tick-1]
            prev_action = np.average(self.normalized_actions[tick-2], keepdims=True)
        # print("NORMALIZED ACTION: ", action)
        # print("NORMALIZED PREV. ACTION: ", prev_action)
        reward = 0
        if np.isnan(state):
            self.without_line += 1
            reward = self.env_config.reward_terminal
        else:
            factors = self.env_config.reward_calculation_weights
            if self.env_config.reward_normalization:
                factors /= np.sum(self.env_config.reward_calculation_weights)

            ratings = self.reward_rating(
                np.array([[state, prev_state], [action, prev_action]]).squeeze(),
                np.array([self.env_config.state_normalization, self.env_config.action_normalization]),
            )

            (state_rating_l, prev_state_rating_l), (action_rating_l, prev_action_rating_l) = ratings['l']
            (state_rating_b, prev_state_rating_b), (action_rating_b, prev_action_rating_b) = ratings['b']
            (state_rating_u, prev_state_rating_u), (action_rating_u, prev_action_rating_u) = ratings['u']

            for factor, component in zip(factors, self.env_config.reward_calculation):
                # print(state_rating_b)
                # print(factor)
                
                # raw (direkt)
                if component == 's': 
                    value = self.reward_raw(state_rating_b)                         # balanced centering (immediately)
                elif component == 'a': 
                    value = self.reward_raw(action_rating_u)                        # upper bound speed  (immediately)
                # deviations (werte innen)
                elif component == 'sd': 
                    value = self.reward_diff(state_rating_b, prev_state_rating_b)   # keep centering (inhibit)
                elif component == 'ad': 
                    value = self.reward_diff(action_rating_u, prev_action_rating_u) # keep speed     (inhibit)
                # improvements (werte außen)
                elif component == 'si': 
                    value = self.reward_comp(state_rating_b, prev_state_rating_b)   # towards center (force)
                elif component == 'ai': 
                    value = self.reward_comp(action_rating_u, prev_action_rating_u) # towards speed  (force)

                reward += np.multiply(factor, value)

            if self.env_config.reward_normalization: 
                norm_range = self.env_config.reward_normalization_range
                min, max = norm_range[0], norm_range[1] 
                reward = normalize(reward, min_range=min, max_range=max)
                # if action[0] == -1.: reward = -1.  # punish stop
            if self.env_config.reward_clipping: 
                reward = clip(reward)

        return reward

    def reward_rating(self, data, anchors):
        # assuming reward is normed between -1 and +1
        # state/action also between [-1, +1] or [0, 1]
        rating = {
            'l': np.divide(
                    np.add(
                        np.multiply(np.add(+anchors[:, [0]], -data), 2),
                        np.ptp(anchors, axis=1, keepdims=True)
                    ),
                    np.ptp(anchors, axis=1, keepdims=True)
            ),
            'b': np.divide(
                    np.multiply(
                        np.add(
                            -np.abs(np.multiply(np.add(-np.mean(anchors, axis=1, keepdims=True), data), 2)), 
                            np.divide(np.ptp(anchors, axis=1, keepdims=True), 2)
                        ), 
                        2
                    ), 
                    np.ptp(anchors, axis=1, keepdims=True)
            ),
            'u': np.divide(
                    np.add(
                        np.multiply(np.add(-anchors[:, [1]], +data), 2), 
                        np.ptp(anchors, axis=1, keepdims=True)), 
                    np.ptp(anchors, axis=1, keepdims=True)
            ),
        }
        
        return rating

    def reward_raw(self, current):
        return current

    def reward_diff(self, current, previous):
        return np.subtract(1, np.abs(np.subtract(previous, current)))

    def reward_comp(self, current, previous):
        return np.divide(np.subtract(current, previous), 2)


def clip(value, min_range=-1., max_range=+1.):
    return np.clip(value, min_range, max_range)


def normalize(value, min_value=-1., max_value=+1., min_range=-1., max_range=+1.):
    return np.add(
        np.multiply(
            np.subtract(max_range, min_range),
            np.divide(
                np.subtract(value, min_value),
                np.subtract(max_value, min_value)
            )
        ),
        min_range
    )


class TwistAction():
    def __init__(self, name, wheel_speeds, separation=.1):
        self.name = name
        self.wheel_speeds = wheel_speeds

        self.action = Twist()
        self.action.linear.x = (wheel_speeds[0] + wheel_speeds[1]) / 2
        self.action.angular.z = (wheel_speeds[0] - wheel_speeds[1]) / separation

        self.stop = Twist()

    def return_instruction(self):
        return self.action

    def return_stop_instruction(self):
        return self.stop

    def __str__(self):
        return f"{self.name}: {self.wheel_speeds[0]}/{self.wheel_speeds[1]}"


class LineFollowingWrapper:
    def __init__(self, step_duration_nsec=100 * 1000 * 1000) -> None:
        self.elapsed_time = time.time_ns()

        self.env_config = EnvironmentConfig()
        self.env_data = EnvironmentData(self.env_config)

        self.step_duration_nsec = step_duration_nsec # the time one RL step() call takes in simulation time

        self.context_count = 0
        self.episode_count = 0
        self.step_count = 0

        self.mode = ""
        self.track = ""
        self.is_training = None
        
        # observation space
        self.channels = self.env_config.processed_features.lower()
        if self.channels in ["bw", "gs"]: channel_size = 1
        elif all(self.channels.find(x) != -1 for x in "rgb"): channel_size = 3
        elif all(self.channels.find(x) != -1 for x in "rgba"): channel_size = 4

        input_shape = [*self.env_config.input_shape, channel_size]
        if self.env_config.sequence_stacking == "v": sequence_dims = np.array([self.env_config.sequence_length, 1, 1])
        elif self.env_config.sequence_stacking == "h": sequence_dims = np.array([1, self.env_config.sequence_length, 1])
        stacked_input_shape = np.multiply(input_shape, sequence_dims)

        self.observation_space = stacked_input_shape

        # action space
        self.action_entries = self.env_config.action_entries
        self.number_actions = len(self.action_entries)

        if not rclpy.ok(): rclpy.init()
        self.manager = LineFollowingManager(self.env_data)
        self.data = LineFollowingData()

        self.info = {
            "mode": self.mode,
            "track": self.track,
            "sequence_length": self.env_config.sequence_length,
            "input_dims": input_shape,
            "number_actions": self.number_actions,
            "counters": self.data.counters
        }

    # -------------> HIGH LEVEL FUNCS

    def close(self):
        Logger().warn('close was called')
        self.manager.switch.publish(EmptyMsg()) # to dump odo data for last track!
        self.manager.destroy_node()
        rclpy.shutdown()

    def switch(self, mode, track):
        self.elapsed_time = time.time_ns()

        start_time = time.time_ns()

        self.env_data.mode = mode
        self.env_data.track = track

        entry = Cache().object_registry['counters'][0]
        entry.append([])
        Cache().update_object('counters', entry)
        
        self.info.update(
            {'mode': self.env_data.mode, 'track': self.env_data.track})

        Logger().warn(f'switch triggered: mode={self.env_data.mode}, track={self.env_data.track}')

        self.context_count += 1; self.data.counters['context'] += 1
        self.episode_count = 0; self.data.counters['episode'] = 0

        name = None
        if (self.env_data.mode == 'eval' and not self.env_config.eval_random_track) or (
            self.env_data.mode == 'train' and not self.env_config.train_random_track):
            name = self.env_data.track

        self.manager.publish_switch(name)

        Evaluator().reinit()

        duration = np.divide(np.diff([start_time, time.time_ns()]), 1e6)
        if self.env_config.debug: Logger().debug(f'switch took [{np.trunc(duration)} ms]')

        self.env_data.switch()

    def reset(self, *, seed=None, options=None):
        self.elapsed_time = time.time_ns()
        start_time = time.time_ns()

        entry = Cache().object_registry['counters'][0]
        if self.data.counters['episode'] > 0:
            entry[-1].append(self.data.counters['tick'])
        Cache().update_object('counters', entry)

        trigger = 'switch' if self.data.counters['episode'] == 0 else 'done'
        Logger().info(f'reset triggered (by {trigger}): counters={self.data.counters}')
        
        self.env_data.reset() # resets some internal vars/counters

        self.episode_count += 1; self.data.counters['episode'] += 1
        self.step_count = 0; self.manager.step = 0; self.data.counters['tick'] = 0
        
        # NOTE: Skipping invalid spawns, until we get a valid one!
        self.state = None
        
        self.manager.trigger_pause(False) # resume sim
        
        # NOTE: AG code block
        self.manager.gz_perform_action_stop()  # send stop action
        self.get_observation(nsec=5e8)  # extend get_obs spinning loop to 0.5s (5e8) or 1s (1e9)
        self.trigger_reset()  # trigger reset after
        img = self.init_step()
        
        # NOTE: trigger_reset if no valid observation can be obtained (for random spawns)
        # while not isinstance(self.state, np.ndarray):
        #     self.trigger_reset()
        #     img = self.init_step()
            
        self.manager.trigger_pause(True) # resume sim

        Evaluator().set_entity(Episode())

        duration = np.divide(np.diff([start_time, time.time_ns()]), 1e6)
        if self.env_config.debug: Logger().debug(f'reset took [{np.trunc(duration)} ms]')

        return img, self.info

    def trigger_reset(self):
        # Get track index/angle
        index = None
        if (self.env_data.mode == 'eval' and not self.env_config.eval_random_position) or (
            self.env_data.mode == 'train' and not self.env_config.train_random_position):
            index = Tracks().index
        angle = None
        if (self.env_data.mode == 'eval' and not self.env_config.eval_random_orientation) or (
            self.env_data.mode == 'train' and not self.env_config.train_random_orientation):
            angle = Tracks().angle

        self.manager.publish_reset(index, angle)

    def init_step(self):
        Logger().info('initial observation request')

        data, _, _, terminated, truncated = self.get_observation(nsec=(500. * 1000. * 1000.))

        # HACK: only set state if valid spawn
        if terminated or truncated: 
            Logger().warn('invalid response, immediately done')
        else:
            self.state = self.handle_observation(data)

        img = self.prepare_img(data) # returns observation for learner model

        return img

    def step(self, action_index:int, randomly_chosen:bool):
        self.manager.trigger_pause(False) # resume sim
        
        quantized_action = self.perform_action(action_index) # perform action

        self.step_count += 1; self.manager.step += 1; self.data.counters['tick'] += 1
        self.env_data.step()

        # get resulting observation
        if self.env_config.debug: Logger().debug('observation request')

        data, quantized_state, clock, terminated, truncated = self.get_observation(nsec=None)

        self.manager.trigger_pause(True)    # pause sim

        self.state = self.handle_observation(data)

        if self.env_config.debug: Logger().debug('reward request')

        reward = self.compute_reward()
        self.reward = self.handle_reward(reward)

        current_time = time.time_ns()
        duration = np.divide(np.abs(np.diff([current_time, self.elapsed_time])), 1e9)
        self.elapsed_time = current_time

        # NOTE: just a hack to save raw action instead of normalized one
        raw_action = self.env_config.action_entries[action_index]
        quantized_action = np.array([raw_action.wheel_speeds[0], raw_action.wheel_speeds[1]])

        Evaluator().set_entity(
            Sample(
                np.array([self.env_data.mode == 'eval'], dtype=np.bool_),
                np.array([randomly_chosen], dtype=np.bool_),
                clock,
                quantized_state,
                quantized_action,
                reward,
                duration
                )
            )

        img = self.prepare_img(data) # returns observation for learner model

        return (img, self.reward, terminated, truncated, self.info)

    # --------------------------------> ACTION

    def perform_action(self, index):
        """ high level action execution """
        if self.env_config.debug: Logger().debug(f'action request at tick [{self.step_count}]')

        action = self.env_config.action_entries[index] # select action to publish to GZ

        self.env_data.actions[self.step_count] = index
        self.env_data.normalized_actions[self.step_count] = self.env_data.normalize_action(self.step_count)
        quantized_action = self.env_data.normalized_actions[self.step_count]
        if self.env_config.debug: 
            Logger().debug(f'action i={index} ({action.wheel_speeds[0]:2.2f}/{action.wheel_speeds[1]:2.2f}) published at tick [{self.step_count}]')

        if self.step_count < 2:
            self.manager.gz_perform_action_stop()
        else:
            self.manager.gz_perform_action(action)

        return quantized_action

    # --------------------------------> OBSERVATION

    def prepare_img(self, img):
        """ convert raw image data into correct shape, transform to grayscale if necessary & normalize [0,1]. """
        obs_dims = (self.observation_space[0], self.observation_space[1], 3)
        out_dims = (1, self.observation_space[0], self.observation_space[1], self.observation_space[2])
        img = np.array(img, dtype=np.float32).reshape(obs_dims)
        
        if self.env_config.processed_features.lower() == 'gs':
            img = np.average(np.divide(img, 255., dtype=np.float32), axis=-1)
            img = np.reshape(img, out_dims)

        return img

    def get_observation(self, nsec=None):
        """ spin for a given time frame to allow collection of data. """
        if nsec is None:
            nsec = self.step_duration_nsec
        s = self.manager.get_last_obs_time()
        # print(s)
        while ((self.manager.get_last_obs_time() - s) < nsec):
            rclpy.spin_once(self.manager, timeout_sec=0)
        response = self.manager.get_data()
        # print(self.manager.get_last_obs_time())
        self.env_data.observations[self.step_count] = np.array(response.data).astype(np.uint8)

        return self.process_observation(response) # NOTE: returns stacked obs + done based on obs. data

    def handle_observation(self, observation):
        return np.average(
                np.divide(
                    observation, 255., dtype=np.float32).reshape(*self.observation_space.shape, -1),
                axis=-1)

    def process_observation(self, response):
        if self.step_count < self.env_config.sequence_length:
            # print(f"{self.env_data.observations[:self.step_count + 1].mean(axis=1, keepdims=True)}")
            temp = np.concatenate([self.env_data.sequence_placeholder, self.env_data.observations[:self.step_count + 1]])[-self.env_config.sequence_length:]
            # print(f"{temp.mean(axis=1, keepdims=True)}")
        else:
            upper = self.step_count + 1
            lower = upper - self.env_config.sequence_length
            temp = self.env_data.observations[lower:upper]
        data = np.flip(temp, axis=0).flatten()

        clock = self.env_data.clocks[self.step_count]
        self.env_data.normalized_states[self.step_count] = self.env_data.normalize_state(self.step_count)
        quantized_state = self.env_data.normalized_states[self.step_count]
        terminated, truncated = self.env_data.check_done(self.env_data.normalized_states[self.step_count])

        return data, quantized_state, clock, terminated, truncated

    # --------------------------------> REWARD

    def compute_reward(self):
        self.env_data.rewards[self.step_count] = self.env_data.calculate_reward(self.step_count)
        # response.improvement = self.env_data.make_assessment(tick)
        return self.env_data.rewards[self.step_count]

    def handle_reward(self, reward):
        return reward[0] # unpacking


class LineFollowingData(metaclass=ThreadSafeSingleton):
    def __init__(self, *args, **kwargs):
        self.mode = ''
        self.track = ''

        self.counters = {
            'context': 0,
            'episode': 0,
            'tick': 0,
        }

        Cache().register_object('counters', [], category='info', trigger='overall')

        Cache().register_object('samples', None, category='debug')
        Cache().register_object('spawns', [], category='debug')

    def counters_array(self):
        return np.array(list(self.counters.values()))


class LineFollowingManager(Node, metaclass=ThreadSafeSingleton):
    def __init__(self, env_data:EnvironmentData, *args, **kwargs):
        super().__init__('linefollowing_manager')

        self.config = Args().args
        self.env_data = env_data
        
        tick_rate_limit = 1e-01
        qos_profile = QoS.QoSProfile(
            history=QoS.HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoS.ReliabilityPolicy.BEST_EFFORT,
            durability=QoS.DurabilityPolicy.VOLATILE,
            deadline=QoS.Duration(seconds=tick_rate_limit),
            lifespan=QoS.Duration(seconds=tick_rate_limit),
        )

        self.step = 0
        self.last_obs_time = 0
        
        self.interval = int(self.config.step_duration_nsec)

        entity_name = '/vehicle'
        world_name = '/world/race_tracks_world'

        # task/episode publishers
        # gives info to other parts of the program that a switch/reset has happened
        self.switch = self.create_publisher(EmptyMsg, f'{world_name}/switch', qos_profile)
        self.reset = self.create_publisher(EmptyMsg, f'{world_name}/reset', qos_profile)

        # receive from camera image topic, send actions to motor topic
        self.gz_observation = self.create_subscription(Image, f'{entity_name}/camera', self.gz_handle_observation_callback, qos_profile)
        self.gz_action = self.create_publisher(Twist, f'{entity_name}/motor', qos_profile)

        # reset world via GZ service
        self.gz_reset = self.create_client(ControlWorld, f'{world_name}/control')
        while not self.gz_reset.wait_for_service(timeout_sec=1.):
            self.get_logger().info('control service unavailable, waiting...')
        self.res_req = ControlWorld.Request()

        # set robot position in gazebo
        self.gz_pose = self.create_client(SetEntityPose, f'{world_name}/set_pose')
        while not self.gz_pose.wait_for_service(timeout_sec=1.):
            self.get_logger().info('position service unavailable, waiting...')
        self.pos_req = SetEntityPose.Request()

        self.gz_clock = self.create_subscription(Clock, f'/clock', self.handle_gz_clock_callback, qos_profile)
        
    def sync_request(self, client, request, wait=None):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if wait is not None: time.sleep(wait)

        return response

    def get_step(self): 
        return self.step

    def get_data(self):
        return self.data

    def gz_handle_observation_callback(self, msg):
        """ gets called whenever a camera message arrives. """
        self.data = msg
        self.last_obs_time = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        # print(f"--- gz obs. cb: {msg.header}")

    def get_last_obs_time(self):
        """ return the last time an observation arrived. """
        return self.last_obs_time

    def gz_perform_action(self, action:TwistAction):
        # action is a TwistAction instance!!
        self.gz_action.publish(action.return_instruction())
        if self.config.debug: Logger().debug(f'action published at step [{self.step}]')

    def gz_perform_action_stop(self):
        action = TwistAction('stop', [0.0, 0.0], 0.1)
        self.gz_action.publish(action.return_stop_instruction()) 
        if self.config.debug: Logger().debug('action stop published')

    def publish_switch(self, name):
        self.switch.publish(EmptyMsg())
        name = Tracks().handle_inter_task_reset(name)

    def publish_reset(self, index, angle):
        self.reset.publish(EmptyMsg())
        
        name = '3pi_robot'
        (position, orientation) = Tracks().handle_intra_task_reset(index, angle)
        spawn_loc = np.concatenate((position, orientation), axis=None)
        entry = Cache().object_registry['spawns'][0]
        if type(entry) == dict:
            entry['pose'] = np.vstack((entry['pose'], spawn_loc))
        else:
            entry = {'pose': spawn_loc}
        Cache().update_object('spawns', entry)
        
        # _ = self.world_control_request()  # NOTE: world reset not working!
        _ = self.set_entity_pose_request(name, position, orientation)  # NOTE: simply move robot!


    def world_control_request(self):
        self.res_req.world_control.reset.all = True
        # self.res_req.world_control.pause = True

        return self.sync_request(self.gz_reset, self.res_req, .1)

    def set_entity_pose_request(self, name, position, orientation):
        self.pos_req.entity.name = name
        self.pos_req.pose.position.x = position[0]
        self.pos_req.pose.position.y = position[1]
        self.pos_req.pose.position.z = position[2]
        self.pos_req.pose.orientation.x = orientation[0]
        self.pos_req.pose.orientation.y = orientation[1]
        self.pos_req.pose.orientation.z = orientation[2]
        self.pos_req.pose.orientation.w = orientation[3]

        return self.sync_request(self.gz_pose, self.pos_req, .1)

    def trigger_pause(self, pause):
        self.res_req.world_control.pause = pause
        # self.gz_reset.call_async(self.res_req)
        self.sync_request(self.gz_reset, self.res_req)
        
        if self.config.debug: Logger().debug(f'pause={pause} request at step [{self.step}]')
        
    def handle_gz_clock_callback(self, msg):
        # print(f"___ gz clock cb: {msg.sim.nanosec}")
        if msg.sim.nanosec % self.interval == 0:
            # print(f"\t msg.nsec: {msg.sim.nanosec}, set interval: {self.interval}")
            # print(f"\t internal step counters: {self.step}, {self.env_data.tick}")
            self.env_data.clocks[self.env_data.tick-1] = np.add(
                [msg.real.sec, msg.sim.sec],
                np.divide([msg.real.nanosec, msg.sim.nanosec], 1e9)
            ).astype(np.float32)
        else:
            pass
            # if self.config.debug: Logger().debug('Skip invalid or redundant message!')        
