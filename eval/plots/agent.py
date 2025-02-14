import os
import sys
import morph
import pprint
import argparse

from utils import helpers


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--custom_1', type=str, default='default_agent_1', help='custom_1 agent help')
parser.add_argument('--custom_2', type=str, default='default_agent_2', help='custom_2 agent help')


TIME_DELTA = 1/2
WHEEL_SEPARATION = 0.1
ACTION_QUANTIZATION = [0.0, 0.25, 0.5]

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


def entry(struct, kwargs):
    helpers.exec(struct, kwargs, return_plots())

def return_plots():
    yield plot_spawn
    yield plot_odometry
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
    labels, data = helpers.extract_debug(['spawn', 'odometry'])

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

    path = os.path.join(os.environ['GIT_PATH'], 'icrl/models/ground_plane/tracks/active.png')
    img = np.array(Image.open(path))

    overlay = np.full_like(img, 0.)
    spawn, odometry = data
    print(spawn.shape)
    print(odometry.shape)
    for offset, current in zip(spawn['pose'], odometry['pose']):
        trajectory = current + offset

        indices = []
        for entry in trajectory:
            index = np.round(m_to_px(entry), 0)
            indices.append(index.astype(int))
        indices = np.stack(indices)
        print('##########')
        print(indices)

        func = interp1d(indices[:, 0], indices[:, 1])
        # y = np.interp(x, indices[:, 0], indices[:, 1])
        overlay[indices[:, 0], indices[:, 1]] = (255, 0, 0, 255)
        for x, diff in zip(indices[:, 0], np.diff(indices[:, 0])):
            print('----------')
            print(x, diff)
            x_pos = np.arange(x + 1, diff)
            y_pos = func(x_pos)
            overlay[x_pos, y_pos] = (0, 0, 255, 255)

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
