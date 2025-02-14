from typing import Optional

import os
import time
import threading
import collections

import numpy as np
import gymnasium as gym

import rclpy
import rclpy.qos as QoS

from rclpy.node import Node

from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty as EmptySrc

from ros_gz_interfaces.srv import ControlWorld, SetEntityPose


from custom_interfaces.srv import *

from rllib_gazebo.utils.ArgsConfig import LearnerArgs

from rllib_gazebo.utils.Evaluation import *
from rllib_gazebo.utils.Logging import Logger
from rllib_gazebo.utils.Caching import Cache
from rllib_gazebo.utils.Tasks import Tracks


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


class LineFollowingManager(Node, metaclass=ThreadSafeSingleton):
    def __init__(self, *args, **kwargs):
        super().__init__('linefollowing_manager')

        self.wait_for_node('simulation_manager', 10)

        tick_rate_limit = 1e-01
        qos_profile = QoS.QoSProfile(
            history=QoS.HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoS.ReliabilityPolicy.BEST_EFFORT,
            durability=QoS.DurabilityPolicy.VOLATILE,
            deadline=QoS.Duration(seconds=tick_rate_limit),
            lifespan=QoS.Duration(seconds=tick_rate_limit),
        )

        rllib_prefix = '/rllib/lf'

        # task/episode publishers
        self.switch = self.create_publisher(EmptyMsg, f'{rllib_prefix}/switch', qos_profile)
        self.switch_msg = EmptyMsg()
        self.reset = self.create_publisher(EmptyMsg, f'{rllib_prefix}/reset', qos_profile)
        self.reset_msg = EmptyMsg()

        # observation client
        self.observation = self.create_client(Observation, f'{rllib_prefix}/observation')
        while not self.observation.wait_for_service(timeout_sec=1.):
            self.get_logger().info('observation service unavailable, waiting...')
        self.obs_req = Observation.Request()

        # action client
        self.action = self.create_client(Action, f'{rllib_prefix}/action')
        while not self.action.wait_for_service(timeout_sec=1.):
            self.get_logger().info('action service unavailable, waiting...')
        self.act_req = Action.Request()

        # reward client
        self.reward = self.create_client(Reward, f'{rllib_prefix}/reward')
        while not self.reward.wait_for_service(timeout_sec=1.):
            self.get_logger().info('reward service unavailable, waiting...')
        self.rew_req = Reward.Request()

        entity_name = '/vehicle'
        world_name = '/world/race_tracks_world'

        self.gz_reset = self.create_client(ControlWorld, f'{world_name}/control')
        while not self.gz_reset.wait_for_service(timeout_sec=1.):
            self.get_logger().info('control service unavailable, waiting...')
        self.res_req = ControlWorld.Request()

        self.gz_pose = self.create_client(SetEntityPose, f'{world_name}/set_pose')
        while not self.gz_pose.wait_for_service(timeout_sec=1.):
            self.get_logger().info('position service unavailable, waiting...')
        self.pos_req = SetEntityPose.Request()

    def sync_request(self, client, request, wait=None):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if wait is not None: time.sleep(wait)

        return response

    def request_observation(self, counter):
        self.obs_req.counter = counter

        # self.trigger_pause(True)
        return self.sync_request(self.observation, self.obs_req)

    def request_action(self, counter, action):
        self.act_req.counter = counter
        self.act_req.index = int(action)

        # self.trigger_pause(False)
        return self.sync_request(self.action, self.act_req)

    def request_reward(self, counter):
        self.rew_req.counter = counter

        return self.sync_request(self.reward, self.rew_req)

    def publish_switch(self, name):
        name = Tracks().handle_inter_task_reset(name)

        self.switch.publish(self.switch_msg)

    def publish_reset(self, index, angle):
        name = '3pi_robot'
        position, orientation = Tracks().handle_intra_task_reset(index, angle)

        entry = Cache().object_registry['spawns'][0]
        try: entry['pose'] = np.stack([entry['pose'], np.array([*position, *orientation])])
        except: entry = {'pose': np.array([*position, *orientation])}
        Cache().update_object('spawns', entry)

        self.world_control_request()
        self.set_entity_pose_request(name, position, orientation)

        self.reset.publish(self.reset_msg)

    def world_control_request(self):
        self.res_req.world_control.reset.all = True

        return self.sync_request(self.gz_reset, self.res_req, 0.1)

    def set_entity_pose_request(self, name, position, orientation):
        self.pos_req.entity.name = name
        self.pos_req.pose.position.x = position[0]
        self.pos_req.pose.position.y = position[1]
        self.pos_req.pose.position.z = position[2]
        self.pos_req.pose.orientation.x = orientation[0]
        self.pos_req.pose.orientation.y = orientation[1]
        self.pos_req.pose.orientation.z = orientation[2]
        self.pos_req.pose.orientation.w = orientation[3]

        return self.sync_request(self.gz_pose, self.pos_req, 0.1)


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
        Cache().register_object('spawns', None, category='debug')

    def counters_array(self):
        return np.array(list(self.counters.values()))


class LineFollowingWrapper(gym.Env[np.ndarray, int]):
    def __init__(self, *args, **kwargs):
        self.elapsed_time = time.time_ns()

        self.config = LearnerArgs().args

        if not rclpy.ok(): rclpy.init()
        self.manager = LineFollowingManager()
        self.data = LineFollowingData()

        self.info = {
            'mode': self.data.mode,
            'track': self.data.track,
            'counters': self.data.counters,
        }

        channels = self.config.processed_features.lower()
        if channels in ['bw', 'gs']: channel_size = 1
        elif all(channels.find(x) != -1 for x in 'rgb'): channel_size = 3
        elif all(channels.find(x) != -1 for x in 'rgba'): channel_size = 4

        input_shape = [*self.config.input_shape, channel_size]
        if self.config.sequence_stacking == 'v':
            sequence_dims = np.array([self.config.sequence_length, 1, 1])
        elif self.config.sequence_stacking == 'h':
            sequence_dims = np.array([1, self.config.sequence_length, 1])
        stacked_input_shape = np.multiply(input_shape, sequence_dims)

        self.observation_space = gym.spaces.Box(0., 1., stacked_input_shape)
        self.action_space = gym.spaces.Discrete(np.prod(self.config.output_shape))

        self.override_flag = False

    def switch(self, mode, track):
        self.elapsed_time = time.time_ns()

        start_time = time.time_ns()

        self.data.mode = mode
        self.data.track = track

        self.info.update({'mode': self.data.mode, 'track': self.data.track})

        entry = Cache().object_registry['counters'][0]
        entry.append([])
        Cache().update_object('counters', entry)

        Logger().warn(f'switch triggered: mode={self.data.mode}, track={self.data.track}')

        self.trigger_switch()
        Evaluator().reinit()

        duration = np.divide(np.diff([start_time, time.time_ns()]), 1e6)
        if self.config.debug: Logger().debug(f'switch took [{np.trunc(duration)} ms]')

    def trigger_switch(self):
        self.data.counters['context'] += 1
        self.data.counters['episode'] = 0

        name = None
        if self.data.mode == 'eval' and not self.config.eval_random_track or self.data.mode == 'train' and not self.config.train_random_track:
            name = self.data.track

        self.manager.publish_switch(name)

    def reset(self, *, seed=None, options=None):
        self.elapsed_time = time.time_ns()

        start_time = time.time_ns()
        super().reset(seed=seed, options=options)

        entry = Cache().object_registry['counters'][0]
        if self.data.counters['episode'] > 0:
            entry[-1].append(self.data.counters['tick'])
        Cache().update_object('counters', entry)

        trigger = 'switch' if self.data.counters['episode'] == 0 else 'done'
        Logger().info(f'reset triggered (by {trigger}): counters={self.data.counters}')

        # HACK: skip invalid spawns, until valid
        self.state = None
        while not isinstance(self.state, np.ndarray):
            self.trigger_reset()
            self.init_step()

        # skip too slow operation, since not needed
        # err_msg = f'{self.state!r} ({type(self.state)}) invalid'
        # assert self.observation_space.contains(self.state), err_msg

        Evaluator().set_entity(Episode())

        duration = np.divide(np.diff([start_time, time.time_ns()]), 1e6)
        if self.config.debug: Logger().debug(f'reset took [{np.trunc(duration)} ms]')

        return (self.state, self.info)

    def trigger_reset(self):
        self.data.counters['episode'] += 1
        self.data.counters['tick'] = 0

        index = None
        if self.data.mode == 'eval' and not self.config.eval_random_position or self.data.mode == 'train' and not self.config.train_random_position:
            index = Tracks().index
        angle = None
        if self.data.mode == 'eval' and not self.config.eval_random_orientation or self.data.mode == 'train' and not self.config.train_random_orientation:
            angle = Tracks().angle

        self.manager.publish_reset(index, angle)

    def init_step(self):
        Logger().info('initial observation request')
        response = self.manager.request_observation(self.data.counters_array())

        # HACK: only set state if valid spawn
        if response.done: Logger().warn('invalid response, immediately done')
        else: self.state = self.handle_observation(response)

    def step(self, action):
        # skip too slow operation, since not needed
        # err_msg = f'{action!r} ({type(action)}) invalid'
        # assert self.action_space.contains(action), err_msg

        # performing the action
        if self.config.debug: Logger().debug('action request')
        response = self.manager.request_action(self.data.counters_array(), action)
        action_quantization = response.quantization

        self.data.counters['tick'] += 1

        # get resulting observation
        if self.config.debug: Logger().debug('observation request')
        response = self.manager.request_observation(self.data.counters_array())
        clock = response.clock
        state_quantization = response.quantization

        self.state = self.handle_observation(response)

        # skip too slow operation, since not needed
        # err_msg = f'{self.state!r} ({type(self.state)}) invalid'
        # assert self.observation_space.contains(self.state), err_msg

        if self.override_flag: self.done = True
        else: self.done = response.done

        self.override_flag = False
        if self.done: Logger().info('episode done')

        # get the calculated reward
        if self.config.debug: Logger().debug('reward request')
        response = self.manager.request_reward(self.data.counters_array())
        # improvement = response.improvement # wird aktuell nicht verwendet
        reward = response.value

        self.reward = self.handle_reward(response)

        current_time = time.time_ns()
        duration = np.divide(np.abs(np.diff([current_time, self.elapsed_time])), 1e9)
        self.elapsed_time = current_time

        # ggf. kritische Zeile?!
        print('duration:', duration)
        Evaluator().set_entity(Sample(np.array([self.data.mode == 'eval'], dtype=np.bool_), np.array([np.nan], dtype=np.bool_), clock, state_quantization, action_quantization, reward, duration))

        return (self.state, self.reward, self.done, False, self.info)

    def handle_observation(self, response):
        # handle channels
        return np.average(np.divide(response.data, 255., dtype=np.float32).reshape(*self.observation_space.shape, -1), axis=-1)

    def handle_reward(self, response):
        # handle unpack
        return response.value[0]

    def override(self):
        Logger().info('ensure this episode ends anyways')
        self.override_flag = True

    def render(self):
        Logger().info('render was called')

    def close(self):
        Logger().info('close was called')

    def destroy(self):
        Logger().warn('destroying the environment')
        self.manager.destroy_node()
        rclpy.shutdown()
