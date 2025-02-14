import os
import sys
import morph
import pprint
import argparse

from utils import helpers


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--custom_1', type=str, default='default_buffer_1', help='custom_1 buffer help')
parser.add_argument('--custom_2', type=str, default='default_buffer_2', help='custom_2 buffer help')


def entry(struct, kwargs):
    helpers.exec(struct, kwargs, return_plots())

def return_plots():
    yield plot_buffer
    yield plot_batch
    yield plot_memorization
    yield plot_utilization


# wie viele samples sind generell in dem buffer
def plot_buffer():
    plot_buffer_by_filter('buffer', [])
    plot_buffer_by_filter('buffer', ['state'])
    plot_buffer_by_filter('buffer', ['action'])
    plot_buffer_by_filter('buffer', ['state', 'action'])

def plot_buffer_by_filter(name, filter):
    labels, data = helpers.extract_data([name])
    # handle filter
    helpers.generic_plot(data, ['_'.join([name, label]) for label in labels])

# wie viele replay samples sind in der batch
def plot_batch():
    plot_batch_by_filter('batch', [])
    plot_batch_by_filter('batch', ['state'])
    plot_batch_by_filter('batch', ['action'])
    plot_batch_by_filter('batch', ['state', 'action'])

def plot_batch_by_filter(name, filter):
    labels, data = helpers.extract_data([name])
    # handle filter
    helpers.generic_plot(data, ['_'.join([name, label]) for label in labels])

def plot_memorization():
    pass

def plot_utilization():
    pass

'''
nochmal für state-quantisierung, von welchem state sind wie viele samples im buffer (geht bei anderen lernparadigmen nicht wirklich)
nochmal für action-quantisierung, von welcher action sind wie viele samples im buffer (von welcher klasse)
und dann für die kombination state-action tupel -> heatmap
'''
