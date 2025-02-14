import os
import sys
import morph
import pprint
import argparse

from utils import helpers


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--custom_1', type=str, default='default_agent_1', help='custom_1 agent help')
parser.add_argument('--custom_2', type=str, default='default_agent_2', help='custom_2 agent help')


TIME_DELTA = 1/6
WHEEL_SEPARATION = 0.1
ACTION_QUANTIZATION = [0.0, 0.25, 0.5]
ACTION_QUANTIZATION = [0.1, 0.3, 0.4, 0.5]

TRACK_TILES = {
    'straight':     ((   0, 5000), (   0,  500)),
    'snake':        ((   0, 5000), ( 500, 1000)),
    'puzzle':       ((   0, 5000), (1000, 1500)),
    'slalom_asym':  ((   0, 5000), (1500, 2000)),
    'slalom_sym':   ((   0, 5000), (2000, 2500)),

    'butterfly_rl': ((   0, 1600), (2500, 5000)),
    'butterfly_lr': ((1600, 3200), (2500, 5000)),
    'jojo_r':       ((3200, 4250), (2500, 3750)),
    'jojo_l':       ((3200, 4250), (3750, 5000)),
    'infinity_rl':  ((4250, 5000), (2500, 3750)),
    'infinity_lr':  ((4250, 5000), (3750, 5000)),
}

TRACK_TILES = {
    'straight': ((0, 5000), (0, 500)),
    'zero_1_l': ((1050, 1350), (2700, 2800)),
    'zero_2_l': ((1050, 1350), (2950, 3050)),
    'zero_3_l': ((1050, 1350), (3200, 3400)),
    'zero_4_l': ((1050, 1350), (3400, 3600)),
    'zero_5_l': ((1050, 1350), (3600, 3850)),
    'zero_6_l': ((1050, 1350), (3850, 4050)),
    'circle_1_l': ((1400, 1600), (2700, 2800)),
    'circle_2_l': ((1400, 1600), (2950, 3050)),
    'circle_3_l': ((1400, 1600), (3200, 3400)),
    'circle_4_l': ((1400, 1600), (3400, 3600)),
    'circle_5_l': ((1400, 1600), (3600, 3850)),
    'circle_6_l': ((1400, 1600), (3850, 4050)),
    'zero_1_r': ((1050, 1350), (2700, 2800)),
    'zero_2_r': ((1050, 1350), (2950, 3050)),
    'zero_3_r': ((1050, 1350), (3200, 3400)),
    'zero_4_r': ((1050, 1350), (3400, 3600)),
    'zero_5_r': ((1050, 1350), (3600, 3850)),
    'zero_6_r': ((1050, 1350), (3850, 4050)),
    'circle_1_r': ((1400, 1600), (2700, 2800)),
    'circle_2_r': ((1400, 1600), (2950, 3050)),
    'circle_3_r': ((1400, 1600), (3200, 3400)),
    'circle_4_r': ((1400, 1600), (3400, 3600)),
    'circle_5_r': ((1400, 1600), (3600, 3850)),
    'circle_6_r': ((1400, 1600), (3850, 4050)),
}


def entry(struct, kwargs):
    helpers.exec(struct, kwargs, return_plots())

def return_plots():
    # yield plot_spawn
    # yield plot_odometry
    yield plot_trajectory


# n = 1 x episode
# shape = (n, 2) pose
# shape = (n, [0], 3) position (coordinates)
# shape = (n, [1], 4) orientation (quaternions)
def plot_spawn():
    labels, data = helpers.extract_debug(['spawn'])
    plot_pose(data, labels)

# n = 2.5 x tick
# shape = (n, 2) odometry
# shape = (n, [0], 2) pose
# shape = (n, [0], [0], 3) position (coordinates)
# shape = (n, [0], [1], 4) orientation (quaternions)
# shape = (n, [1], 2) twist
# shape = (n, [1], [0], 3) linear (velocity)
# shape = (n, [1], [1], 3) angular (velocity)
def plot_odometry():
    labels, data = helpers.extract_debug(['odometry'])
    plot_pose(data, labels)
    plot_twist(data, labels)

# use odo data from sim + spawns => store both in debug
def plot_trajectory():
    import numpy as np
    from PIL import Image
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation

    def quaternion_to_euler(quaternions):                       # expects a 4-shaped array (quaternion)
        temp = Rotation.from_quat(quaternions)
        return temp.as_euler(seq='XYZ', degrees=True)           # extrinsic rotation 'XYZ'

    def euler_to_quaternion(angle):
        temp = Rotation.from_euler('XYZ', angle, degrees=True)
        return temp.as_quat()

    def m_to_px(pos): # expects single data point
        pos_x, pos_y = +pos[0], -pos[1]                         # negate Y
        factor = np.divide((5000, 5000), (100, 100))            # image -> plane

        temp_x = np.multiply(np.add(pos_y, 50), factor[0])      # 1. add displacement; 2. factorize
        temp_y = np.multiply(np.add(pos_x, 50), factor[1])      # invert axis

        return np.array([temp_x, +temp_y]).T                    # transpose to convert from shape [2,N] -> [N,2]

    def px_to_m(pos):
        pos_x, pos_y = +pos[0], +pos[1]
        factor = np.divide((100, 100), (5000, 5000))            # plane -> image

        temp_x = np.add(np.multiply(pos_y, factor[1]), -50)     # invert axis
        temp_y = np.add(np.multiply(pos_x, factor[0]), -50)     # 1. factorize; 2. add displacement

        return np.array([temp_x, -temp_y]).T                    # negate Y

    # path = os.path.join(os.environ['GIT_PATH'], 'icrl/models/ground_plane/tracks/active.png')
    path = os.path.join(os.environ['GIT_PATH'], 'icrl/models/ground_plane/tracks/test.png')
    img = np.array(Image.open(path))

    labels, data = helpers.extract_debug(['spawn', 'odometry'])
    counters = helpers.DATA['info']['counters']
    print('len(counters):', len(counters))

    overlay = np.full_like(img, 0.)
    spawn, odometry = data
    print('len(spawn["pose"]):', len(spawn['pose']))
    print('len(odometry["pose"]):', len(odometry['pose']))

    data_color = (255,   0,   0, 255)
    data_color = (  0,   0, 255, 255)
    fill_color = (  0, 255,   0, 255)

    last = 0
    for episodes, offsets, locations in zip(counters, spawn['pose'], odometry['pose']):
        print('len(episodes):', len(episodes))
        print('sum(episodes):', sum(episodes))

        print('offsets.shape:', offsets.shape)
        print('locations.shape:', locations.shape)

        # offsets += np.array([0, 0, 0, *euler_to_quaternion([0, 0, 90])])

        print(offsets[0])
        print(locations[0])
        print(locations[0] - last)

        '''
        index = 0
        for episode, offset in zip(episodes, offsets):
            rescale = int(np.floor(episode * (len(locations) / sum(episodes))))
            # rescale = int(np.round(episode * (len(locations) / sum(episodes))))
            # rescale = int(np.ceil(episode * (len(locations) / sum(episodes))))

            lower = index
            upper = lower + rescale
            index = upper

            # temp = locations[upper - 1].copy()
            # print(temp)
            locations[lower:upper] -= last
            locations[lower:upper] += offset
            # last = temp
            # print(last)

            last = locations[upper - 1] - offset
            # last = locations[upper - 1] + last
        '''

        trajectory = np.squeeze(locations)

        indices = []
        for entry in trajectory:
            index = np.round(m_to_px(entry), 0)
            indices.append(index.astype(int))
        indices = np.stack(indices)

        print('np.min(indices[:, 0]):', np.min(indices[:, 0]))
        print('np.min(indices[:, 1]):', np.min(indices[:, 1]))
        print('np.max(indices[:, 0]):', np.max(indices[:, 0]))
        print('np.max(indices[:, 1]):', np.max(indices[:, 1]))

        lower_check = [0, 0]
        upper_check = [5000, 5000]
        indices = indices[np.all(indices > lower_check, axis=1) & np.all(indices < upper_check, axis=1)]

        if data_color == (255, 0, 0, 255): data_color = (0, 0, 255, 255)
        if data_color == (0, 0, 255, 255): data_color = (255, 0, 0, 255)

        if len(indices) > 0:
            func = interp1d(indices[:, 0], indices[:, 1])
            # y = np.interp(x, indices[:, 0], indices[:, 1])
            overlay[indices[:, 0], indices[:, 1]] = data_color
            # for x, diff in zip(indices[:, 0], np.diff(indices[:, 0])):
            #     for x_diff in np.arange(diff):
            #         x_pos = x + x_diff
            #         y_pos = func(x_pos)
            #         x_pos = np.round(x_pos, 0).astype(int)
            #         y_pos = np.round(y_pos, 0).astype(int)
            #         # if x_pos < 0 and x_pos > 5000: continue
            #         # if y_pos < 0 and y_pos > 5000: continue
            #         overlay[x_pos, y_pos] = fill_color

        # ACHTUNG: zwecks debugging eingerückt!!!
    # =>
    img += overlay

    fig = helpers.V.new_figure(size=(19.2, 12))
    plt = helpers.V.new_plot(fig, align='h', axes='2d')
    helpers.V.imshow(plt, img)
    helpers.V.generate(fig)

def plot_pose(data, labels):
    data = data['pose']
    labels = helpers.labels(labels, suffix='pose')

    plot_position(data, labels)
    plot_orientation(data, labels)

def plot_position(data, labels):
    data = data[:, :, [0, 1, 2]]
    labels = helpers.labels(labels, suffix='position')
    helpers.generic_plot(data, labels)

def plot_orientation(data, kwargs, labels):
    data = data[:, :, [3, 4, 5, 6]]
    labels = helpers.labels(labels, suffix='orientation')
    helpers.generic_plot(data, labels)

def plot_twist(data, labels):
    data = data['twist']
    labels = helpers.labels(labels, suffix='twist')

    plot_linear(data, labels)
    plot_angular(data, labels)

def plot_linear(data, labels):
    data = data[:, :, [0, 1, 2]]
    labels = helpers.labels(labels, suffix='linear')
    helpers.generic_plot(data, labels)

def plot_angular(data, labels):
    data = data[:, :, [3, 4, 5]]
    labels = helpers.labels(labels, suffix='angular')
    helpers.generic_plot(data, labels)

'''
Trajectory Visualization:
    Plot the agent's trajectories in the environment over different episodes.
    This showcases how the agent explores and exploits the environment and how its behavior changes with learning.
'''
