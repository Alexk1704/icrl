import os
import sys
import time
import itertools
import threading
import numpy as np

import rclpy
import rclpy.qos as QoS

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty as EmptySrc

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from ros_gz_interfaces.msg import Clock
from ros_gz_interfaces.srv import ControlWorld


from custom_interfaces.srv import *

from rllib_gazebo.utils.ArgsConfig import SimulationArgs
from rllib_gazebo.utils.Exchanging import Exchanger
from rllib_gazebo.utils.Logging import Logger


class SimulationManagerLF(Node):
    def __init__(self) -> None:
        super().__init__('simulation_manager')

        self.wait_for_node('ros_gz_bridge', 10)

        self.elapsed = None

        self.initialized = threading.Event()
        self.initialized.clear()

        self.available = threading.Event()
        self.available.clear()

        tick_rate_limit = 1e-01
        qos_profile = QoS.QoSProfile(
            history=QoS.HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoS.ReliabilityPolicy.BEST_EFFORT,
            durability=QoS.DurabilityPolicy.VOLATILE,
            deadline=QoS.Duration(seconds=tick_rate_limit),
            lifespan=QoS.Duration(seconds=tick_rate_limit),
        )

        group_1 = MutuallyExclusiveCallbackGroup()
        group_2 = MutuallyExclusiveCallbackGroup()
        group_3 = MutuallyExclusiveCallbackGroup()
        group_4 = MutuallyExclusiveCallbackGroup()
        group_5 = MutuallyExclusiveCallbackGroup()

        rllib_prefix = '/rllib/lf'

        # task/episode
        self.rllib_switch = self.create_subscription(EmptyMsg, f'{rllib_prefix}/switch', self.handle_rllib_switch_callback, qos_profile, callback_group=group_1)
        self.rllib_reset = self.create_subscription(EmptyMsg, f'{rllib_prefix}/reset', self.handle_rllib_reset_callback, qos_profile, callback_group=group_1)

        # observation, action, reward
        self.rllib_observation = self.create_service(Observation, f'{rllib_prefix}/observation', self.handle_rllib_observation_callback, callback_group=group_2)
        self.rllib_action = self.create_service(Action, f'{rllib_prefix}/action', self.handle_rllib_action_callback, callback_group=group_2)
        self.rllib_reward = self.create_service(Reward, f'{rllib_prefix}/reward', self.handle_rllib_reward_callback, callback_group=group_2)

        entity_name = '/vehicle'
        world_name = '/world/race_tracks_world'

        # gazebo
        self.gz_reset = self.create_client(ControlWorld, f'{world_name}/control', callback_group=group_3)
        while not self.gz_reset.wait_for_service(timeout_sec=1.):
            self.get_logger().info('control service unavailable, waiting...')
        self.res_req = ControlWorld.Request()

        # fires way to often, therefore isolated...
        self.gz_clock = self.create_subscription(Clock, f'/clock', self.handle_gz_clock_callback, qos_profile, callback_group=group_4)

        self.gz_observation = self.create_subscription(Image, f'{entity_name}/camera', self.handle_gz_observation_callback, qos_profile, callback_group=group_5)
        self.gz_action = self.create_publisher(Twist, f'{entity_name}/motor', qos_profile, callback_group=group_5)

        config = EnvironmentConfig()
        self.data = EnvironmentData(config)

        self.counter = 0
        self.last_msg = None

        self.frequency = 2
        self.interval = int(1e+09 // self.frequency)

        '''
        add a callback queue once again
        trigger callback (pause/unpause request) first
        uncouple the current callback logic and trigger it subsequently
        '''

        # isn't possible to call an callback in another callback even for an MT executor
        # outsourcing the pausing functionality even to the environment makes no sense
        # therefore, we use a second node, which only spins if this function is triggered

    def handle_continue_sim(self):
        self.initialized.set()
        self.available.clear()

        if self.data.config.force_waiting:
            self.trigger_pause(False)

    def handle_rllib_switch_callback(self, msg):
        start_time = time.time_ns()

        self.data.switch()
        self.handle_continue_sim()

        duration = np.divide(np.diff([start_time, time.time_ns()]), 1e6)
        if self.data.config.debug: Logger().debug(f'switch took [{np.trunc(duration)} ms]')

    def handle_rllib_reset_callback(self, msg):
        start_time = time.time_ns()

        self.data.reset()
        self.handle_continue_sim()

        duration = np.divide(np.diff([start_time, time.time_ns()]), 1e6)
        if self.data.config.debug: Logger().debug(f'reset took [{np.trunc(duration)} ms]')

    def handle_counter_check(self, request_counter, wait_until_ready):
        actual_counter = np.array([self.data.context, self.data.episode, self.data.tick])

        if wait_until_ready:
            while np.any(np.not_equal(request_counter, actual_counter)):
                Logger().info(f'waiting for sync to prevent mismatch')
                if self.data.config.debug: Logger().debug(f'actual_counter={actual_counter}, request_counter={request_counter}')

                self.initialized.wait()
                self.initialized.clear()

                actual_counter = np.array([self.data.context, self.data.episode, self.data.tick])

        # TODO: must be thrown on the main thread to be displayed
        assert np.all(np.equal(request_counter, actual_counter)), f'counter mismatch: {request_counter} (frontend) != {actual_counter} (backend)'

    def handle_rllib_action_callback(self, request, response):
        context, episode, tick = request.counter
        if self.data.config.debug: Logger().debug(f'action request at tick [{tick}]')

        self.handle_counter_check(request.counter, False)

        self.data.actions[tick] = request.index
        self.data.normalized_actions[tick] = self.data.normalize_action(tick)
        response.quantization = self.data.normalized_actions[tick]

        self.handle_gz_action(request.index)

        return response

    def handle_rllib_observation_callback(self, request, response):
        context, episode, tick = request.counter
        if self.data.config.debug: Logger().debug(f'observation request at tick [{tick}]')

        self.handle_counter_check(request.counter, True)

        # muss das ggf. vor das assert?
        # macht eigentlich keinen Sinn, oder?
        # warte doch, wenn ein reset kommt haben wir zwei mal hintereinander ein obs-request
        # also kein step von der action, sondern ein reset vom step, aber der wurde bereits vom assert gecheckt?

        # wobei der reset code ja eigentlich wartet
        # und der erste request erst danach erfolgt
        # nur der reset hier ist async + zweifach gestaffelt
        # lediglich der sim reset + replace wartet 2 sekunden
        self.available.wait()
        self.available.clear()

        # observation sequence
        if tick < self.data.config.sequence_length:
            temp = np.concatenate([self.data.sequence_placeholder, self.data.observations[:tick + 1]])[-self.data.config.sequence_length:]
        else:
            upper = tick + 1
            lower = upper - self.data.config.sequence_length
            temp = self.data.observations[lower:upper]
        response.data = np.flip(temp, axis=0).flatten()

        response.clock = self.data.clocks[tick]
        self.data.normalized_states[tick] = self.data.normalize_state(tick)
        response.quantization = self.data.normalized_states[tick]
        response.done = self.data.check_done(self.data.normalized_states[tick])

        if response.done:
            Logger().info('Episode is done!')

        return response

    def handle_rllib_reward_callback(self, request, response):
        context, episode, tick = request.counter
        if self.data.config.debug: Logger().debug(f'reward request at tick [{tick}]')

        self.handle_counter_check(request.counter, False)

        self.data.rewards[tick] = self.data.calculate_reward(tick)
        response.improvement = self.data.make_assessment(tick)
        response.value = self.data.rewards[tick]

        return response

    def trigger_pause(self, pause):
        self.res_req.world_control.pause = pause

        # NOTE: muss hier gewartet werden oder wie ist die pipeline?!
        self.gz_reset.call_async(self.res_req)
        if self.data.config.debug: Logger().debug(f'pause={pause} request at tick [{self.data.tick}]')

    def handle_gz_clock_callback(self, msg):
        # header does not exist
        if msg.sim.nanosec % self.interval == 0:
            self.data.clocks[self.data.tick] = np.add(
                [msg.real.sec, msg.sim.sec],
                np.divide([msg.real.nanosec, msg.sim.nanosec], 1e9)
            ).astype(np.float32)
        else:
            pass
            # if self.data.config.debug: Logger().debug('Skip invalid or redundant message!')

    def handle_gz_observation_callback(self, msg):
        if msg.header.stamp.nanosec % self.interval == 0:
            if self.data.config.force_waiting: self.trigger_pause(True)
            if self.data.config.debug: Logger().debug(f'observation subscribed at tick [{self.data.tick}]')
            self.data.observations[self.data.tick] = np.array(msg.data).astype(np.uint8)
            self.available.set()
        else:
            if self.data.config.debug: Logger().debug('Skip invalid or redundant message!')

    def handle_gz_action(self, index):
        action = self.data.config.action_entries[index]
        self.gz_action.publish(action.return_instruction())
        if self.data.config.debug: Logger().debug(f'action published at tick [{self.data.tick}]')
        if self.data.config.force_waiting: self.trigger_pause(False)

        self.data.step()

class TwistAction():
    def __init__(self, name, wheel_speeds, separation=.1):
        self.name = name
        self.wheel_speeds = wheel_speeds

        self.action = Twist()
        self.action.linear.x = (wheel_speeds[0] + wheel_speeds[1]) / 2
        self.action.angular.z = (wheel_speeds[0] - wheel_speeds[1]) / separation

    def return_instruction(self):
        return self.action

    def to_string(self):
        return f'{self.name}: {self.wheel_speeds}'


class EnvironmentConfig():
    def __init__(self) -> None:
        for name, value in vars(SimulationArgs().args).items():
            setattr(self, name, value)

        # ---

        self.observation_dims = {k: v for k, v in zip('xyz', self.state_shape)}
        self.desired_line_center = np.divide(self.observation_dims['y'], 2)

        # ---

        # use a differently procured spaces (non-linear)
        # margins with less values as centers
        start, stop, number = self.state_quantization
        self.state_space = np.linspace(start, stop, int(number))

        start, stop, number = self.action_quantization
        self.action_space = itertools.product(
            np.linspace(start, stop, int(number)),
            np.linspace(start, stop, int(number)),
        )

        self.action_entries = []
        for wheel_speeds in self.action_space:
            diff = np.diff(wheel_speeds)
            if diff == 0: self.action_entries.append(TwistAction('forward', wheel_speeds))
            elif diff < 0: self.action_entries.append(TwistAction('right', wheel_speeds))
            elif diff > 0: self.action_entries.append(TwistAction('left', wheel_speeds))


class EnvironmentData():
    def __init__(self, config:EnvironmentConfig) -> None:
        self.config = config

        '''
        the variable "context" only counts the number of swaps
        always store episodes, use glob for some logrotate
        differ between train and eval for storing?!
        '''
        self.context = 0
        self.episode = 0
        self.tick = 0

        self.without_line = 0

        self.stats = {
            'length': {'last': 0., 'curr': 0., 'acc': 0., 'avg': 0., 'min': 0., 'max': 0.},
            'return': {'last': 0., 'curr': 0., 'acc': 0., 'avg': 0., 'min': 0., 'max': 0.},
        }

        '''
        self.max_history = 100
        handle all of them as cyclic arrays (ring buffers) and store before they are overwritten
        '''
        self.storage_location = os.path.join(self.config.root_dir, self.config.exp_id, 'debug', 'Raw')

        '''
        attention, these arrays are never reset or even cleaned, e.g., by overriding with zeros (would not be that inefficient)
        only their global index is dropped and not longer tracked, but all values will still remain in this case
        '''
        self.observations = np.empty((self.config.max_steps_per_episode, np.prod(list(self.config.observation_dims.values()))), np.uint8)
        self.actions = np.empty((self.config.max_steps_per_episode, 1), np.uint8)
        self.rewards = np.empty((self.config.max_steps_per_episode, 1), np.float32)

        self.clocks = np.empty((self.config.max_steps_per_episode, 2), np.float32)
        self.normalized_states = np.empty((self.config.max_steps_per_episode, 1), np.float32)
        self.normalized_actions = np.empty((self.config.max_steps_per_episode, 2), np.float32)
        self.normalized_rewards = np.empty((self.config.max_steps_per_episode, 1), np.float32)

        self.sequence_placeholder = np.zeros((self.config.sequence_length, self.observations.shape[1]), np.uint8)

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
        if self.config.debug: Logger().debug(f'switch invoked')

        if self.context > 0:
            self.update_stats()

            if self.config.debug:
                Logger().debug(f'stats={self.stats}')

        self.context += 1
        self.episode = 0

        self.reinit_stats()

    def reset(self):
        if self.config.debug: Logger().debug(f'reset invoked')

        if self.episode > 0:
            self.update_stats()

            if self.config.debug:
                Logger().debug(f'stats={self.stats}')

        self.episode += 1
        self.tick = 0

        self.without_line = 0

    def step(self):
        self.tick += 1

    def prepare_observation(self, tick):
        observation = self.observations[tick]
        rgb_image = observation.reshape(*self.config.observation_dims.values())
        grayscale_image = np.average(np.divide(rgb_image, 255., dtype=np.float32), axis=-1)

        return grayscale_image

    def prepare_action(self, tick):
        index = self.actions[tick]
        action = self.config.action_entries[index[0]]

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
        if self.config.line_mode == 'c': state = self.quantize_line_position(image)
        elif self.config.line_mode == 'l': state = self.quantize_line_position(image, 'l')
        elif self.config.line_mode == 'r': state = self.quantize_line_position(image, 'r')

        if not np.isnan(state):
            return normalize(state, 0, self.config.observation_dims['y'], *self.config.state_normalization)
        return np.nan

    def normalize_action(self, tick):
        action = self.prepare_action(tick)

        return normalize(action.wheel_speeds, *self.config.action_quantization[:2], *self.config.action_normalization)

    def quantize_line_position(self, image:np.ndarray, mode:str='c', simple:bool=True):
        if simple: image = np.mean(image, axis=0, keepdims=True)

        mask = (image < self.config.line_threshold)

        if np.all(~mask): return np.nan
        # if np.all(mask): return np.nan

        # l_white_index = 0 + np.mean(np.argmin(mask, axis=-1))
        l_black_index = 0 + np.mean(np.argmax(mask, axis=-1))

        mask = np.flip(mask, axis=-1)

        # r_white_index = self.config.observation_dims['y'] - np.mean(np.argmin(mask, axis=-1)) -1
        r_black_index = self.config.observation_dims['y'] - np.mean(np.argmax(mask, axis=-1)) -1

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
        if np.isnan(state): return True
        if not self.tick < self.config.max_steps_per_episode: return True
        if not self.without_line < self.config.max_steps_without_line: return True
        return False

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

        state = self.normalized_states[tick]
        action = np.average(self.normalized_actions[tick], keepdims=True)

        if tick == 1:
            prev_state = state
            prev_action = action
        else:
            prev_state = self.normalized_states[tick - 1]
            prev_action = np.average(self.normalized_actions[tick - 1], keepdims=True)

        reward = 0
        if np.isnan(state):
            self.without_line += 1
            reward = self.config.reward_terminal
        else:
            factors = self.config.reward_calculation_weights
            if self.config.reward_normalization:
                factors /= np.sum(self.config.reward_calculation_weights)

            '''
            ggf. unten im Code on-the-fly berechnen
            lässt sich leider nicht via einer Pipeline ausrechnen?!
            wobei ggf. einmalig rechnen und andere bound nur Vorzeichen flip und balanced = /2
            '''
            # calculate all once and pass the corresponding values to each function
            # (state_rating, prev_state_rating), (action_rating, prev_action_rating) = self.reward_rating(
            #     np.array([[state, prev_state], [action, prev_action]]).squeeze(),
            #     np.array([self.config.state_normalization, self.config.action_normalization]),
            # )

            ratings = self.reward_rating(
                np.array([[state, prev_state], [action, prev_action]]).squeeze(),
                np.array([self.config.state_normalization, self.config.action_normalization]),
            )

            (state_rating_l, prev_state_rating_l), (action_rating_l, prev_action_rating_l) = ratings['l']
            (state_rating_b, prev_state_rating_b), (action_rating_b, prev_action_rating_b) = ratings['b']
            (state_rating_u, prev_state_rating_u), (action_rating_u, prev_action_rating_u) = ratings['u']

            # set reward components also only once in init
            # reduce the overheat to a minimum if static called
            # this loop is ridiculous, really!!! -_-
            for factor, component in zip(factors, self.config.reward_calculation):
                '''
                anders handhaben, da aktuell absolut egal ist, welcher Wert für die diff/comp Berechnung einfließt
                was hat weniger Overhead einmalig davor berechnen vs nur das, was benötigt wird
                '''
                # raw (direkt)
                if component == 's': value = self.reward_raw(state_rating_b)                            # balanced centering (immediately)
                elif component == 'a': value = self.reward_raw(action_rating_u)                         # upper bound speed  (immediately)
                # deviations (werte innen)
                elif component == 'sd': value = self.reward_diff(state_rating_b, prev_state_rating_b)   # keep centering (inhibit)
                elif component == 'ad': value = self.reward_diff(action_rating_u, prev_action_rating_u) # keep speed     (inhibit)
                # improvements (werte außen)
                elif component == 'si': value = self.reward_comp(state_rating_b, prev_state_rating_b)   # towards center (force)
                elif component == 'ai': value = self.reward_comp(action_rating_u, prev_action_rating_u) # towards speed  (force)

                # if self.config.reward_normalization: value = normalize(value)
                # if self.config.reward_clipping: value = clip(value)

                reward += np.multiply(factor, value)

            if self.config.reward_normalization: reward = normalize(reward)
            if self.config.reward_clipping: reward = clip(reward)

        return reward

    def reward_rating(self, data, anchors):
        # assuming reward is normed between -1 and +1
        # state/action also between [-1, +1] or [0, 1]
        rating = {
            'l': np.divide(np.add(np.multiply(np.add(+anchors[:, [0]], -data), 2), np.ptp(anchors, axis=1, keepdims=True)), np.ptp(anchors, axis=1, keepdims=True)),
            'b': np.divide(np.multiply(np.add(-np.abs(np.multiply(np.add(-np.mean(anchors, axis=1, keepdims=True), data), 2)), np.divide(np.ptp(anchors, axis=1, keepdims=True), 2)), 2), np.ptp(anchors, axis=1, keepdims=True)),
            'u': np.divide(np.add(np.multiply(np.add(-anchors[:, [1]], +data), 2), np.ptp(anchors, axis=1, keepdims=True)), np.ptp(anchors, axis=1, keepdims=True)),
        }

        # if self.reward_normalization:
        #     rating = {key: normalize(value) for key, value in rating.items()}

        # if self.reward_clipping:
        #     rating = {key: clip(value) for key, value in rating.items()}

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


def main():
    if not rclpy.ok(): rclpy.init()
    executor = MultiThreadedExecutor()
    node = SimulationManagerLF()

    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPT')
    finally:
        Logger().del_async()
        # sys.exit(0)
        # os._exit(0)

    executor.remove_node(node)
    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    # explicitly init exchanger and logger
    Exchanger()
    Logger()

    main()
