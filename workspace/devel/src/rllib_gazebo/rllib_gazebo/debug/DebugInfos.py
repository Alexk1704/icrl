import os
import time
import pickle
import pprint
import argparse

import numpy as np

import rclpy
import rclpy.qos as QoS

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty as EmptySrc

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from ros_gz_interfaces.msg import Clock


from custom_interfaces.srv import *


parser = argparse.ArgumentParser()
parser.add_argument('--odometry',    nargs=3, type=int, default=[0, 5, -1],    help='')
parser.add_argument('--clock',       nargs=3, type=int, default=[0, 1000, -1], help='')
parser.add_argument('--image',       nargs=3, type=int, default=[0, 2, -1],    help='')
parser.add_argument('--twist',       nargs=3, type=int, default=[0, 0, -1],    help='')
parser.add_argument('--observation', nargs=3, type=int, default=[0, 0, -1],    help='')
parser.add_argument('--action',      nargs=3, type=int, default=[0, 0, -1],    help='')
parser.add_argument('--reward',      nargs=3, type=int, default=[0, 0, -1],    help='')
args = parser.parse_args()


class DebugInfoSim(Node):
    def __init__(self, config) -> None:
        super().__init__('debug_manager')

        self.wait_for_node('simulation_manager', 10)

        self.lookups = {}
        if config.odometry[0]: self.register_lookup(Odometry, 'msg', config.odometry[1], config.odometry[2], attributes=['pose', 'twist'], exclude='covariance')
        if config.clock[0]: self.register_lookup(Clock, 'msg', config.clock[1], config.clock[2], attributes=['sim_time', 'real_time'])
        if config.image[0]: self.register_lookup(Image, 'msg', config.image[1], config.image[2], attributes=['data'])
        if config.twist[0]: self.register_lookup(Twist, 'msg', config.twist[1], config.twist[2], attributes=['linear', 'angular'])
        if config.observation[0]: self.register_lookup(Observation, 'srv', config.observation[1], config.observation[2], attributes=['data'])
        if config.action[0]: self.register_lookup(Action, 'srv', config.action[1], config.action[2], attributes=['index'])
        if config.reward[0]: self.register_lookup(Reward, 'srv', config.reward[1], config.reward[2], attributes=['value'])

        tick_rate_limit = 1e-01
        qos_profile = QoS.QoSProfile(
            history=QoS.HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoS.ReliabilityPolicy.BEST_EFFORT,
            durability=QoS.DurabilityPolicy.VOLATILE,
            deadline=QoS.Duration(seconds=tick_rate_limit),
            lifespan=QoS.Duration(seconds=tick_rate_limit),
        )

        # to be still threadsafe
        group_1 = MutuallyExclusiveCallbackGroup()
        group_2 = MutuallyExclusiveCallbackGroup()
        group_3 = MutuallyExclusiveCallbackGroup()
        group_4 = MutuallyExclusiveCallbackGroup()

        rllib_prefix = '/rllib/lf'

        # task/episode
        self.rllib_switch = self.create_subscription(EmptyMsg, f'{rllib_prefix}/switch', self.handle_rllib_switch_callback, qos_profile, callback_group=group_1)
        self.rllib_reset = self.create_subscription(EmptyMsg, f'{rllib_prefix}/reset', self.handle_rllib_reset_callback, qos_profile, callback_group=group_1)

        entity_name = '/vehicle'
        world_name = '/world/race_tracks_world'

        # subscribers
        if Odometry in self.lookups:
            self.odometry_subscriber = self.create_subscription(Odometry, f'{entity_name}/odo', self.handle_gz_debug_callback(Odometry), qos_profile, callback_group=group_2)
        if Clock in self.lookups:
            self.clock_subscriber = self.create_subscription(Clock, '/clock', self.handle_gz_debug_callback(Clock), qos_profile, callback_group=group_3)
        if Image in self.lookups:
            self.image_subscriber = self.create_subscription(Image, f'{entity_name}/camera', self.handle_gz_debug_callback(Image), qos_profile, callback_group=group_3)
        if Twist in self.lookups:
            self.twist_subscriber = self.create_subscription(Twist, f'{entity_name}/motor', self.handle_gz_debug_callback(Twist), qos_profile, callback_group=group_3)

        # services
        if Observation in self.lookups:
            self.observation_service = self.create_service(Observation, f'{rllib_prefix}/observation', self.handle_gz_debug_callback(Observation), callback_group=group_4)
        if Action in self.lookups:
            self.action_service = self.create_service(Action, f'{rllib_prefix}/action', self.handle_gz_debug_callback(Action), callback_group=group_4)
        if Reward in self.lookups:
            self.reward_service = self.create_service(Reward, f'{rllib_prefix}/reward', self.handle_gz_debug_callback(Reward), callback_group=group_4)

        # FIXME: HARDCODED PATH
        self.path = os.path.join(os.environ['DAT_PATH'], os.environ['EXP_ID'], 'debug')

        for entry in self.lookups:
            path = os.path.join(self.path, entry.__name__)
            if not os.path.exists(path):
                os.makedirs(path)

        self.cnt = {}
        self.bulk = {}
        self.temp = {}

        self.details = False

    def handle_rllib_switch_callback(self, msg):
        self.dump_infos('switch')

    def handle_rllib_reset_callback(self, msg):
        self.dump_infos('reset')

    def handle_gz_debug_callback(self, msg_type):
        lookup = self.lookups[msg_type]

        if lookup['frequency']['topic'] > 0:
            topic_interval = int(np.divide(1e+09, lookup['frequency']['topic']))
        else:
            topic_interval = lookup['frequency']['topic']

        if lookup['frequency']['capture'] > 0:
            capture_interval = int(np.divide(1e+09, lookup['frequency']['capture']))
        else:
            capture_interval = lookup['frequency']['capture']

        attributes = lookup['attributes']

        def closure_skip(msg):
            if self.details: print(msg.header)

        def closure_each(msg):
            if self.details: print(msg.header)
            self.collect_infos(msg, attributes)

        def closure_filter(msg):
            if msg.header.stamp.nanosec % capture_interval < topic_interval:
                if self.details: print(msg.header)
                self.collect_infos(msg, attributes)

        if capture_interval == 0:
            return closure_skip
        elif capture_interval == -1:
            return closure_each
        else:
            return closure_filter

    def cleanup_infos(self, entry):
        self.bulk[entry] = {}
        for attribute in self.lookups[entry]['attributes']:
            self.bulk[entry].update({attribute: []})

    def collect_infos(self, msg, attributes):
        try:
            msg_type = type(msg)
            for attribute in attributes:
                if self.details: print(getattr(msg, attribute))
                self.bulk[msg_type][attribute].append(getattr(msg, attribute))
        except KeyError: pass

    def resolve_infos(self, info, filter):
        include = filter.get('include', [])
        exclude = filter.get('exclude', [])

        temp = []
        try:
            for attribute in info.get_fields_and_field_types():
                if (not include or attribute in include) and (not exclude or attribute not in exclude):
                    if self.details: print(getattr(info, attribute))

                    try: temp.extend(self.resolve_infos(getattr(info, attribute), filter))
                    except: temp.append(self.resolve_infos(getattr(info, attribute), filter))
        except AttributeError:
            try: temp.extend(info)
            except: temp.append(info)
        return temp

    def simplify_infos(self, infos, filter):
        for key, values in infos.items():
            data = [self.resolve_infos(value, filter) for value in values]
            infos[key] = np.array(data)
        return infos

    def dump_infos(self, entity):
        try: self.cnt[entity]
        except: self.cnt[entity] = -1

        for entry in self.lookups:
            if self.lookups[entry]['trigger'] == entity:
                try: self.temp = self.bulk[entry].copy()
                except: self.temp.clear()
                self.cleanup_infos(entry)

                infos = self.simplify_infos(self.temp, self.lookups[entry]['filter'])
                if infos:
                    file_name = os.path.join(self.path, entry.__name__, f'{entity}_{self.cnt[entity]}.pkl')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(infos, fp)

        self.cnt[entity] += 1

    def register_lookup(self, msg_type, topic_type, topic_freq=-1, capture_freq=-1, attributes=[], include=[], exclude=[], trigger='switch'):
        self.lookups.update({
            msg_type: {
                'type': topic_type,
                'frequency': {
                    'topic': topic_freq,
                    'capture': capture_freq,
                },
                'attributes': attributes,
                'filter': {
                    'include': include,
                    'exclude': exclude,
                },
                'trigger': trigger,
            }
        })


def main():
    if not rclpy.ok(): rclpy.init()
    executor = MultiThreadedExecutor()
    node = DebugInfoSim(args)

    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPT')

    executor.remove_node(node)
    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
