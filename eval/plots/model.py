import os
import sys
import morph
import pprint
import argparse

from utils import helpers
# from cl_replay.api.utils.plot import vis_protos # NOTE: sets mpl backend to agg


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--custom_1', type=str, default='default_model_1', help='custom_1 model help')
parser.add_argument('--custom_2', type=str, default='default_model_2', help='custom_2 model help')


def entry(struct, kwargs):
    helpers.exec(struct, kwargs, return_plots())

def return_plots():
    yield plot_protos
    yield plot_time
    yield plot_learning_rate
    yield plot_batch_size
    yield plot_parameters
    yield plot_activation
    yield plot_prediction
    yield plot_target
    yield plot_gradient
    yield plot_log
    yield plot_loss
    yield plot_accuracy
    yield plot_forgetting
    yield plot_confusion


def plot_protos():
    options = {
        'epoch': -1,
        'proto_size': [12, 100],
        'channels': 1,
    }
    exp_id = data['info']['exp_id']
    path = data['info']['paths']['protos']
    save_dir = os.path.join(kwargs['path'], 'protos', exp_id)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    protos_paths = [] # aggregate all protos for each task/track
    track_protos = os.listdir(path)
    for i, protos in enumerate(sorted(track_protos)):
        track_protos = os.path.join(path, protos)
        prefix = protos[:(protos.find('_protos') + 1)] + 'L2_GMM_'
        prefix_path = os.path.join(track_protos, 'E0', prefix)
        save_path = os.path.join(save_dir, protos)
        vis_protos.vis_img_data(prefix=prefix, prefix_path=prefix_path, suffix=0, out=save_path, **options)


def plot_time():
    labels, data = helpers.extract_data(['time'])
    helpers.generic_plot(data, labels)

def plot_learning_rate():
    labels, data = helpers.extract_data(['learning_rate'])
    helpers.generic_plot(data, labels)

def plot_batch_size():
    labels, data = helpers.extract_data(['batch_size'])
    helpers.generic_plot(data, labels)

def plot_parameters():
    labels, data = helpers.extract_data(['parameter'])
    helpers.generic_plot(data, labels)

def plot_activation():
    labels, data = helpers.extract_data(['activations'])
    helpers.generic_plot(data, labels)

def plot_prediction():
    labels, data = helpers.extract_data(['prediction'])
    helpers.generic_plot(data, labels)

def plot_target():
    labels, data = helpers.extract_data(['target'])
    helpers.generic_plot(data, labels)

def plot_gradient():
    labels, data = helpers.extract_data(['gradient'])
    helpers.generic_plot(data, labels)

def plot_log():
    labels, data = helpers.extract_data(['log'])
    helpers.generic_plot(data, labels)

def plot_loss():
    labels, data = helpers.extract_data(['loss'])
    helpers.generic_plot(data, labels)

def plot_accuracy():
    labels, data = helpers.extract_data(['accuracy'])
    helpers.generic_plot(data, labels)

def plot_forgetting():
    labels, data = helpers.extract_data(['forgetting'])
    helpers.generic_plot(data, labels)

def plot_confusion():
    labels, data = helpers.extract_data(['confusion'])
    helpers.generic_plot(data, labels)

'''
confusion matrix:
    print a heatmap of the actions (taken ones and "correct/optimal" ones -> most rewarding)
    not possible, as there are no "real" labels, the optimal action can only be inferred over time

correct/rectify or improve/enhance matrix:
    if samples are replayed/offline training and it is no longer a decision making process, rather than a simple prediction problem
    here, we can visualize the rate of differing actions, where the current one changes compared to the original one
'''
