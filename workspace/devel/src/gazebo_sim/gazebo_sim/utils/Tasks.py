import os
import numpy as np
import random

from PIL import Image as img
from scipy.spatial.transform import Rotation


class Tracks():
    def __new__(cls):
        try: return cls.__instance__
        except:
            cls.initialize_once()
            cls.__instance__ = super().__new__(cls)
            return cls.__instance__

    @classmethod
    def initialize_once(cls):
        # FIXME: HARDCODED PATH
        # path = os.path.join(os.environ['GIT_PATH'], 'icrl/models/ground_plane/tracks/active.png')
        path = os.path.join(os.environ['GIT_PATH'], 'icrl/models/ground_plane/tracks/test.png')

        cls.track = np.array(img.open(path))[:, :, -1]

        cls.track_mask = np.not_equal(cls.track, 0)
        cls.track_indices = np.argwhere(cls.track_mask)
        
        cls.tiles = {
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
        
        cls.entries = {
            'straight': [
                {'index': (4976, 251), 'angle': 0},
            ],
            'zero_1_l': [
                {'index': (1245, 2751), 'angle': 0},
            ],
            'zero_2_l': [
                {'index': (1240, 3001), 'angle': 0},
            ],
            'zero_3_l': [
                {'index': (1235, 3251), 'angle': 0},
            ],
            'zero_4_l': [
                {'index': (1225, 3501), 'angle': 0},
            ],
            'zero_5_l': [
                {'index': (1200, 3751), 'angle': 0},
            ],
            'zero_6_l': [
                {'index': (1175, 4001), 'angle': 0},
            ],
            'circle_1_l': [
                {'index': (1500, 2751), 'angle': 0},
            ],
            'circle_2_l': [
                {'index': (1500, 3001), 'angle': 0},
            ],
            'circle_3_l': [
                {'index': (1500, 3251), 'angle': 0},
            ],
            'circle_4_l': [
                {'index': (1500, 3501), 'angle': 0},
            ],
            'circle_5_l': [
                {'index': (1500, 3751), 'angle': 0},
            ],
            'circle_6_l': [
                {'index': (1500, 4001), 'angle': 0},
            ],
            'zero_1_r': [
                {'index': (1245, 2751), 'angle': 180},
            ],
            'zero_2_r': [
                {'index': (1245, 3001), 'angle': 180},
            ],
            'zero_3_r': [
                {'index': (1245, 3251), 'angle': 180},
            ],
            'zero_4_r': [
                {'index': (1245, 3501), 'angle': 180},
            ],
            'zero_5_r': [
                {'index': (1245, 3751), 'angle': 180},
            ],
            'zero_6_r': [
                {'index': (1245, 4001), 'angle': 180},
            ],
            'circle_1_r': [
                {'index': (1500, 2751), 'angle': 180},
            ],
            'circle_2_r': [
                {'index': (1500, 3001), 'angle': 180},
            ],
            'circle_3_r': [
                {'index': (1500, 3251), 'angle': 180},
            ],
            'circle_4_r': [
                {'index': (1500, 3501), 'angle': 180},
            ],
            'circle_5_r': [
                {'index': (1500, 3751), 'angle': 180},
            ],
            'circle_6_r': [
                {'index': (1500, 4001), 'angle': 180},
            ],
        }
        
        # cls.tiles = {
        #     'straight': ((0, 5000), (0, 500)),
        #     'snake': ((0, 5000), (500, 1000)),
        #     'puzzle': ((0, 5000), (1000, 1500)),
        #     'slalom_asym': ((0, 5000), (1500, 2000)),
        #     'slalom_sym': ((0, 5000), (2000, 2500)),

        #     'butterfly_rl': ((0, 1600), (2500, 5000)),
        #     'butterfly_lr': ((1600, 3200), (2500, 5000)),
        #     'jojo_r': ((3200, 4250), (2500, 3750)),
        #     'jojo_l': ((3200, 4250), (3750, 5000)),
        #     'infinity_rl': ((4250, 5000), (2500, 3750)),
        #     'infinity_lr': ((4250, 5000), (3750, 5000)),
        # }
        
        # cls.entries = {
        #     'straight': [
        #         {'index': (4976, 251), 'angle': 0},
        #         {'index': (26, 251), 'angle': 180},
        #     ],
        #     'snake': [
        #         {'index': (4976, 712), 'angle': 270},
        #         {'index': (26, 712), 'angle': 90},
        #     ],
        #     'puzzle': [
        #         {'index': (4976, 1290), 'angle': 0},
        #         {'index': (28, 1290), 'angle': 180},
        #     ],
        #     'slalom_asym': [
        #         {'index': (4976, 1751), 'angle': 270},
        #         {'index': (22, 1751), 'angle': 90},
        #     ],
        #     'slalom_sym': [
        #         {'index': (4976, 2251), 'angle': 270},
        #         {'index': (27, 2251), 'angle': 90},
        #     ],
        #     'butterfly_rl': [
        #         {'index': (802, 3306), 'angle': 0},
        #         {'index': (833, 4196), 'angle': 180},
        #     ],
        #     'butterfly_lr': [
        #         {'index': (2400, 3306), 'angle': 180},
        #         {'index': (2369, 4196), 'angle': 0},
        #     ],
        #     'jojo_r': [
        #         {'index': (3726, 2626), 'angle': 0},
        #         {'index': (3735, 3626), 'angle': 180},
        #     ],
        #     'jojo_l': [
        #         {'index': (3726, 3876), 'angle': 180},
        #         {'index': (3715, 4876), 'angle': 0},
        #     ],
        #     'infinity_rl': [
        #         {'index': (4626, 2839), 'angle': 240},
        #         {'index': (4626, 3413), 'angle': 90},
        #     ],
        #     'infinity_lr': [
        #         {'index': (4626, 4089), 'angle': 300},
        #         {'index': (4626, 4663), 'angle': 90},
        #     ],
        # }

        cls.info = {
            'image': {
                'base': ((0, -1), (1, 0)),
                'size': (5000, 5000),
                'origin': (0, 0),
                'tilt': (0, 0, 0),
            },
            'plane': {
                'base': ((0, 1), (-1, 0)),
                'size': (100, 100),
                'origin': (-50, -50),
                'tilt': (0, 0, 90),
            }
        }

        cls.attempts = [0, 0, 0]

        cls.retry_bound = 3
        cls.retry_method = [cls.respawn, cls.reset, cls.replace]

    @classmethod
    def respawn(cls):
        # new (random) position
        # This method simply resets the robot to the initial starting position every time it goes off course.
        # While this method is simple, it doesn't necessarily encourage the robot to learn from its mistakes or recover from errors.
        pass

    @classmethod
    def reset(cls):
        # latest/nearest checkpoint
        # This method uses checkpoints as a reference on the track and resets the robot to the last checkpoint it passed before going off course.
        # This method encourages the robot to learn to navigate to the next checkpoint, but it may not be suitable for tasks where the checkpoints are too far apart.
        pass

    @classmethod
    def replace(cls):
        # latest/nearest valid position
        # This method resets the robot to the last position on the track where it was following the line.
        # This can be an effective way of encouraging the robot to learn from its off-course behavior and correct its course while still allowing it to make progress towards the end of the track.
        pass

    @classmethod
    def perform_lookup(cls, name):
        # use lower -1 and upper +1
        # lower included, upper excluded
        (x_lower, x_upper), (y_lower, y_upper) = cls.tiles[name]
        track_tile = cls.track[x_lower:x_upper, y_lower:y_upper]

        local_mask = np.not_equal(track_tile, 0)
        local_indices = np.argwhere(local_mask)

        global_mask = np.zeros((5001, 5001))
        global_mask[x_lower:x_upper, y_lower:y_upper] = local_mask
        global_indices = local_indices + [x_lower, y_lower]

        cls.mask = global_mask
        cls.indices = global_indices

        cls.index = cls.entries[name][0]['index']
        cls.angle = cls.entries[name][0]['angle']

        cls.min_x = np.min(global_indices[:, 0])
        cls.min_y = np.min(global_indices[:, 1])
        cls.max_x = np.max(global_indices[:, 0])
        cls.max_y = np.max(global_indices[:, 1])

        cls.x_diff = np.abs(np.ptp(global_indices[:, 0]))
        cls.y_diff = np.abs(np.ptp(global_indices[:, 1]))

        cls.min_x_by_min_y = np.min(global_indices[global_indices[:, 1] == cls.min_y][:, 0])
        cls.min_x_by_max_y = np.min(global_indices[global_indices[:, 1] == cls.max_y][:, 0])
        cls.max_x_by_min_y = np.max(global_indices[global_indices[:, 1] == cls.min_y][:, 0])
        cls.max_x_by_max_y = np.max(global_indices[global_indices[:, 1] == cls.max_y][:, 0])
        cls.min_y_by_min_x = np.min(global_indices[global_indices[:, 0] == cls.min_x][:, 1])
        cls.min_y_by_max_x = np.min(global_indices[global_indices[:, 0] == cls.max_x][:, 1])
        cls.max_y_by_min_x = np.max(global_indices[global_indices[:, 0] == cls.min_x][:, 1])
        cls.max_y_by_max_x = np.max(global_indices[global_indices[:, 0] == cls.max_x][:, 1])

        # construct NE, SE, SW, NW from source
        # min_x_min_y -> NW
        # min_x_max_y -> NE
        # max_x_min_y -> SW
        # max_x_max_y -> SE
        # or alternatively switch to N, E, S, W by averaging?!

        # drop these, they are either not the global extrema or invalid positions
        # cls.min_xy = np.min(global_indices, axis=0)
        # cls.max_xy = np.max(global_indices, axis=0)

        # cls.min_xy_by_min_x = np.min(global_indices[global_indices[:, 0] == cls.min_xy[0]], axis=0)
        # cls.min_xy_by_max_x = np.min(global_indices[global_indices[:, 0] == cls.max_xy[0]], axis=0)
        # cls.min_xy_by_min_y = np.min(global_indices[global_indices[:, 1] == cls.min_xy[1]], axis=0)
        # cls.min_xy_by_max_y = np.min(global_indices[global_indices[:, 1] == cls.max_xy[1]], axis=0)

        # but this can be empty...
        # cls.min_xy_by_min_x_and_min_y = np.min(global_indices[global_indices[:, 0] == cls.min_x and global_indices[:, 1] == cls.min_y], axis=0)
        # cls.min_xy_by_min_x_and_max_y = np.min(global_indices[global_indices[:, 0] == cls.min_x and global_indices[:, 1] == cls.max_y], axis=0)
        # cls.min_xy_by_max_x_and_min_y = np.min(global_indices[global_indices[:, 0] == cls.max_x and global_indices[:, 1] == cls.min_y], axis=0)
        # cls.min_xy_by_max_x_and_max_y = np.min(global_indices[global_indices[:, 0] == cls.max_x and global_indices[:, 1] == cls.max_y], axis=0)
        # cls.max_xy_by_min_x_and_min_y = np.max(global_indices[global_indices[:, 0] == cls.min_x and global_indices[:, 1] == cls.min_y], axis=0)
        # cls.max_xy_by_min_x_and_max_y = np.max(global_indices[global_indices[:, 0] == cls.min_x and global_indices[:, 1] == cls.max_y], axis=0)
        # cls.max_xy_by_max_x_and_min_y = np.max(global_indices[global_indices[:, 0] == cls.max_x and global_indices[:, 1] == cls.min_y], axis=0)
        # cls.max_xy_by_max_x_and_max_y = np.max(global_indices[global_indices[:, 0] == cls.max_x and global_indices[:, 1] == cls.max_y], axis=0)

        '''
        find default indices and angles
        marking the start/end of the track
        '''

        '''
        # tight box
        if cls.x_diff < cls.y_diff:
            alignment = 'W'
            direction = 'E'
        if cls.x_diff == cls.y_diff:
            alignment = 'S'
            direction = 'N'
        if cls.x_diff > cls.y_diff:
            alignment = 'S'
            direction = 'N'

        position = {
            'N': [np.mean(), np.mean()],
            'E': [np.mean(), np.mean()],
            'S': [np.mean(), np.mean()],
            'W': [np.mean(), np.mean()],
        }

        orientation = {
            'N': 0,
            'E': 270,
            'S': 180,
            'W': 90,
        }

        cls.index = position[alignment]
        cls.angle = orientation[direction]
        '''

        '''
        # bounding box
        if np.abs(np.diff([cls.min_xy[0], cls.max_xy[0]])) < np.abs(np.diff([cls.min_xy[1], cls.max_xy[1]])):
            alignment = 'W'
            direction = 'E'
        if np.abs(np.diff([cls.min_xy[0], cls.max_xy[0]])) == np.abs(np.diff([cls.min_xy[1], cls.max_xy[1]])):
            alignment = 'S'
            direction = 'N'
        if np.abs(np.diff([cls.min_xy[0], cls.max_xy[0]])) > np.abs(np.diff([cls.min_xy[1], cls.max_xy[1]])):
            alignment = 'S'
            direction = 'N'

        position = {
            'N': cls.max_xy,
            'E': cls.max_xy,
            'S': cls.max_xy,
            'W': cls.max_xy,
        }

        orientation = {
            'N': 0,
            'E': 270,
            'S': 180,
            'W': 90,
        }

        cls.index = position[alignment]
        cls.angle = orientation[direction]
        '''

        '''
        N = 100
        indices = cls.indices.copy()
        np.random.shuffle(indices)
        plt.scatter(indices[:N, 1], indices[:N, 0], 10, ['yellow'])

        start = cls.entries[name][0]['index']
        end = cls.entries[name][1]['index']

        plt.scatter(start[1], start[0], 100, ['green'])
        plt.scatter(end[1], end[0], 100, ['red'])
        '''

        '''
        # bounding box approach
        # or its tight version (dual)
        print('index:', cls.index)
        print('angle:', cls.angle)

        plt.scatter(cls.index[1], cls.index[0], 100, ['white'])
        '''

    @classmethod
    def pixel_to_meter(cls, index):
        factor = np.divide(cls.info['plane']['size'], cls.info['image']['size'])
        temp_x = np.add(np.multiply(index[1], factor[1]), cls.info['plane']['origin'][1])
        temp_y = np.add(np.multiply(index[0], factor[0]), cls.info['plane']['origin'][0])
        position = np.array([+temp_x, -temp_y, 0.])
        # print('index:', index)
        # print('position:', position)

        return position

    @classmethod
    def euler_to_quaternion(cls, angle):
        temp = Rotation.from_euler('xyz', [0, 0, angle], degrees=True)
        orientation = temp.as_quat(canonical=False)
        # print('angle:', angle)
        # print('orientation:', orientation)

        return orientation

    @classmethod
    def estimate_name(cls):
        return np.random.choice(list(cls.tiles.keys()))

    @classmethod
    def estimate_index(cls):
        # pos_align=None, rot_align=None, mapping='pixel'
        return cls.indices[np.random.randint(len(cls.indices))]

    @classmethod
    def estimate_angle(cls, index):
        # pos_align=None, rot_align=None, mapping='pixel'
        if np.array_equal(index, cls.index):
            return np.add(np.add(cls.angle, np.random.uniform(-30, +30)), +90)  # just orientation

        scope = [10, 10]
        valid_range = 30  # if both, random position & orientation

        lower_bound = np.clip(np.subtract(index, scope), [0, 0], cls.info['image']['size'])
        upper_bound = np.clip(np.add(index, scope), [0, 0], cls.info['image']['size'])

        focus_block = cls.mask[lower_bound[0]:upper_bound[0] + 1, lower_bound[1]:upper_bound[1] + 1]

        directions = {}
        if np.abs(np.diff([cls.min_x, index[0]])) <= np.abs(np.diff([cls.max_x, index[0]])):
            directions.update({'S': focus_block[len(focus_block) - 1, :]})
        else:
            directions.update({'N': focus_block[0, :]})

        if np.abs(np.diff([cls.min_y, index[1]])) <= np.abs(np.diff([cls.max_y, index[1]])):
            directions.update({'E': focus_block[:, len(focus_block) - 1]})
        else:
            directions.update({'W': focus_block[:, 0]})

        angles = []
        for name, direction in directions.items():
            if np.count_nonzero(direction) > 0:
                if name in ['N', 'S']:
                    dists = np.subtract(np.argwhere(direction), scope[1])
                elif name in ['W', 'E']:
                    dists = np.subtract(np.argwhere(direction), scope[0])

                opposite = dists[np.argmin(np.abs(dists))]
                adjacent = scope

                angle = np.degrees(np.arctan(np.divide(opposite, adjacent)))

                if name == 'N': angle = -angle + 0
                elif name == 'W': angle = -angle + 90
                elif name == 'S': angle = +angle + 180
                elif name == 'E': angle = +angle + 270

                angles.extend(angle)

        if len(angles) > 0:
            angle = np.random.choice(angles)
        else:
            angle = np.random.randint(360)

        angle += np.random.uniform(-valid_range, +valid_range)

        return angle

    @classmethod
    def handle_inter_task_reset(cls, name=None):
        cls.counter = 0

        if name is None: name = cls.estimate_name()
        cls.perform_lookup(name)

        return name

    @classmethod
    def handle_intra_task_reset(cls, index=None, angle=None):
        if index is None: index = cls.estimate_index()
        if angle is None: angle = cls.estimate_angle(index)
        else: angle += 90 # HACK

        position = cls.pixel_to_meter(index)
        orientation = cls.euler_to_quaternion(angle)

        return (position, orientation)
'''
import matplotlib as mpl
import matplotlib.pyplot as plt

T = Tracks()
for name in T.tiles:
    T.perform_lookup(name)

plt.imshow(T.track)
plt.show()
'''
'''
T = Tracks()
for key, value in T.entries.items():
    print('track:', key)
    for name, entry in zip(['start', 'end'], value):
        print(f'  - {name}')

        index, angle = entry['index'], entry['angle']
        angle += 90

        meter = T.pixel_to_meter(index)
        quaternion = T.euler_to_quaternion(angle)

        print('\tposition:      x: {:6.2f},   y: {:6.2f},   z: {:6.2f}'.format(*meter))
        print('\torientation:   x: {:6.2f},   y: {:6.2f},   z: {:6.2f}'.format(*[0, 0, angle]))
'''