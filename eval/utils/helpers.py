import os
import sys
import morph
import pprint
import itertools
import functools

import numpy as np
import pandas as pd
import multiprocessing as mp

from utils import tools
from utils.Analyzer import Analyzer
from utils.Visualizer import Visualizer

from rllib_gazebo.utils.Evaluation import *


LOOKUPS = {
    'wrapper': Wrapper,
    'history': History,
    'trace': Trace,
    'episode': Episode,
    'step': Step,
    'sample': Sample,
}

A = Analyzer()
V = Visualizer()

DATA = None
KWARGS = None


def exec(struct, kwargs, plots):
    if kwargs['draft'] == 'yes': V.change_options('Qt5Agg', 'fast')

    if kwargs['exec'] == 'st':
        if kwargs['debug'] == 'yes': print('concurrent execution disabled')
        st_exec(struct, kwargs, plots)
    if kwargs['exec'] == 'mt':
        if kwargs['debug'] == 'yes': print('concurrent execution enabled')
        mt_exec(struct, kwargs, plots)

def st_exec(struct, kwargs, plots):
    global DATA, KWARGS
    wrapper = generate_iterable(struct, kwargs)

    KWARGS = kwargs
    for entry in wrapper:
        DATA = entry
        # map(invoke, plots)
        for plot in plots:
            plot()

def mt_exec(struct, kwargs, plots):
    global DATA, KWARGS
    wrapper = generate_iterable(struct, kwargs)

    KWARGS = kwargs
    for entry in wrapper:
        DATA = entry
        with mp.Pool() as pool:
            pool.map(invoke, plots)

def invoke(func):
    func()

def create(kwargs, fig=None, axs=None):
    if kwargs['draft'] == 'no': V.change_options('Qt5Cairo', kwargs['style'])
    if fig is None: fig = V.new_figure(title=kwargs['title'], size=kwargs['size'])
    if axs is None: axs = V.new_plot(fig, title='', axes=[kwargs['detail'], ''], align=kwargs['align'], legend=True)
    return fig, axs

def finish(kwargs, fig=None, axs=None):
    if fig is None: fig = V.current_figure()
    if axs is None: axs = V.current_plot()

    path = None
    if kwargs['mode'] == 'save':
        dirname = os.path.join(kwargs['path'], kwargs['module'])
        filename = '.'.join(kwargs['name'], kwargs['format'])
        path = os.path.join(dirname, filename)
        tools.ensure_path(path)
    V.generate(fig, path)

def generic_plot(data, labels, fig=None, axs=None):
    func, data = check_data(data)
    if func is not None and data is not None:
        fig, axs = create(KWARGS, fig, axs)
        func(data, labels, KWARGS, fig, axs)
        finish(KWARGS, fig, axs)

# 1-d data vs n-d data
# discrete vs continuous data
def content_1(data, labels, kwargs, fig, axs):
    for i, entry in enumerate(data):
        if i > 0 and kwargs['plot'] == 'multiple':
            fig, axs = create(kwargs, fig, None)
        if kwargs['category'] == 'categorical':
            if kwargs['type'] == '': V.pie_1D(axs, labels, entry)
            elif 'pie' in kwargs['type']: V.pie_1D(axs, labels, entry)
            elif 'bar' in kwargs['type']: V.bar_2D(axs, labels, None, entry)
        if kwargs['category'] == 'distribution':
            if kwargs['type'] == '': V.hist_1D(axs, labels, entry)
            elif 'hist' in kwargs['type']: V.hist_1D(axs, labels, entry)
            elif 'box' in kwargs['type']: V.box_1D(axs, labels, entry)
            elif 'violin' in kwargs['type']: V.violin_1D(axs, labels, entry)
        if kwargs['category'] == 'timeseries':
            if kwargs['type'] == '': V.curve_2D(axs, labels, None, entry)
            elif 'curve' in kwargs['type']: V.curve_2D(axs, labels, None, entry)
            elif 'scatter' in kwargs['type']: V.scatter_2D(axs, labels, None, entry)
            elif 'stem' in kwargs['type']: V.stem_2D(axs, labels, None, entry)
            elif 'step' in kwargs['type']: V.step_2D(axs, labels, None, entry)

def content_2(data, labels, kwargs, fig, axs):
    for i, entry in enumerate(data):
        if i > 0 and kwargs['plot'] == 'multiple':
            fig, axs = create(kwargs, fig, None)
        if kwargs['category'] == 'categorical':
            if kwargs['type'] == '': pass
        if kwargs['category'] == 'distribution':
            if kwargs['type'] == '': pass
            elif 'hist' in kwargs['type']: V.hist_2D(axs, labels, entry[:, 0], entry[:, 1])
            elif 'hexa' in kwargs['type']: V.hexa_2D(axs, labels, entry[:, 0], entry[:, 1])
        if kwargs['category'] == 'timeseries':
            if kwargs['type'] == '': V.curve_2D(axs, labels, entry[:, 0], entry[:, 1])
            elif 'curve' in kwargs['type']: V.curve_2D(axs, labels, entry[:, 0], entry[:, 1])
            elif 'scatter' in kwargs['type']: V.scatter_2D(axs, labels, entry[:, 0], entry[:, 1])
            elif 'stem' in kwargs['type']: V.stem_2D(axs, labels, entry[:, 0], entry[:, 1])
            elif 'step' in kwargs['type']: V.step_2D(axs, labels, entry[:, 0], entry[:, 1])

def content_3(data, labels, kwargs, fig, axs):
    for i, entry in enumerate(data):
        if i > 0 and kwargs['plot'] == 'multiple':
            fig, axs = create(kwargs, fig, None)
        entry = A.clip(entry, -2, +1000)
        print(A.stats(entry))
        # labels, entry = np.unique(entry, return_counts=True)
        # V.pie_1D(axs, labels, entry)
        V.hist_1D(axs, labels, entry.astype(np.float32))
        fig, axs = create(kwargs, fig, None)
        V.violin_1D(axs, labels, entry.astype(np.float32), orientation='h')
        fig, axs = create(kwargs, fig, None)
        V.curve_2D(axs, labels, None, entry)
        V.scatter_2D(axs, labels, None, entry)
        if kwargs['category'] == 'timeseries':
            if kwargs['type'] == '': pass
            # size
            elif 'scatter' in kwargs['type']: V.scatter_2D
            elif 'scatter' in kwargs['type']: V.scatter_3D
        if kwargs['category'] == 'heatmap':
            if kwargs['type'] == '': pass
            elif 'heatmap' in kwargs['type']: V.heatmap_v2_2D
            elif 'contour' in kwargs['type']: V.tricontour_2D
        if kwargs['category'] == 'surface':
            if kwargs['type'] == '': pass
            elif 'surface' in kwargs['type']: V.trisurf_3D
            elif 'contour' in kwargs['type']: V.tricontour_3D

def content_4(data, labels, kwargs, fig, axs):
    for i, entry in enumerate(data):
        if i > 0 and kwargs['plot'] == 'multiple':
            fig, axs = create(kwargs, fig, None)

def generate_iterable(data_struct, kwargs):
    # Assumptions:
    # - Each run has the same track sequence
    # - Each run has the same backend model
    for eval_id in data_struct:
        for wrap_id in data_struct[eval_id]:
            for comb_id in data_struct[eval_id][wrap_id]:
                for run_id in data_struct[eval_id][wrap_id][comb_id]:
                    if kwargs['debug'] == 'yes': debug(data_struct[eval_id][wrap_id][comb_id][run_id])
                    yield data_struct[eval_id][wrap_id][comb_id][run_id]
                    # print(sub_task.keys())
                    # print({key: pd.DataFrame.from_records(value) for key, value in sub_task.items()})
                    # yield {key: pd.DataFrame(value) for key, value in sub_task.items()}

def debug(data):
    def resolve_data(data, level=0):
        ident = ' ' * level
        if isinstance(data, dict):
            for key, value in data.items():
                print(f'{ident}{key}')
                resolve_data(value, level + 2)
        else:
            print(f'{ident}type:', type(data))
            if isinstance(data, np.ndarray): print(f'{ident}shape:', data.shape)
            elif isinstance(data, list): print(f'{ident}length:', len(data))
            else: print(f'{ident}value:', data)

    for entry in data['raw']:
        resolve_data(entry)

# extract all needed data only once
# than use lookups to access the already extracted data
# no logic on the level of these modules
def unpack(data):
    wrapper = {}
    for entry in data['raw']:
        for key in entry:
            for _key in entry[key]:
                try: wrapper[_key].append(entry[key][_key])
                except: wrapper[_key] = [entry[key][_key]]

                name = _key
                name = name.replace('eval', '')
                name = name.replace('train', '')
                if name != _key:
                    try: wrapper[name].append(entry[key][_key])
                    except: wrapper[name] = [entry[key][_key]]
    return wrapper

def extract(data, entities, ignore_mode=True):
    wrapper = {}
    for entity in entities:
        for entry in data:
            if entry.find(entity) != -1:
                name = entry
                if ignore_mode:
                    name = name.replace('eval', '')
                    name = name.replace('train', '')
                wrapper[name] = data[entry]
    return wrapper

def extract_sample(entities, ignore_mode=True):
    wrapper = extract(unpack(DATA), [''], ignore_mode)

    raw = []
    for entry in wrapper['samples']:
        raw.extend(resolve_samples_as_list(entry, level='sample'))
    wrapper = build_samples(raw)
    wrapper = group_samples(wrapper, 'extend')

    wrapper = extract(wrapper, entities, ignore_mode)

    if len(wrapper) == 0:
        labels = None
        stacked = None
    else:
        labels = list(wrapper.keys())
        stacked = np.stack(list(wrapper.values()), axis=0)

    return labels, stacked

def extract_data(entities, ignore_mode=True):
    wrapper = extract(unpack(DATA), entities, ignore_mode)

    temp = [0, 0, 0]
    for key, value in wrapper.items():
        value = np.concatenate(value)
        if len(value.shape) == 1:
            value = np.expand_dims(value, axis=1)
        if len(value.shape) == 2:
            value = np.expand_dims(value, axis=1)
        wrapper[key] = value
        temp = np.max([temp, value.shape], axis=0)

    # resolve stacking and concating hierarchy
    # like mosaic/tiling issue
    for key, value in wrapper.items():
        template = np.full(temp, np.nan).flatten()
        template[:np.prod(value.shape)] = value.flatten()
        average = np.average(template.reshape(temp), axis=1)
        try: wrapper[key] = np.squeeze(average, axis=-1)
        except: wrapper[key] = average

    if len(wrapper) == 0:
        labels = None
        stacked = None
    else:
        labels = list(wrapper.keys())
        stacked = np.stack(list(wrapper.values()), axis=1)

    return labels, stacked

def extract_debug(entities, ignore_mode=True):
    wrapper = extract(unpack(DATA), entities, ignore_mode)

    if len(wrapper) == 0:
        labels = None
        stacked = None
    else:
        labels = list(wrapper.keys())
        stacked = [{key: np.concatenate([entry[key].reshape(-1, 1, entry[key].shape[-1]) for entry in value]) for key in value[0].keys()} for value in wrapper.values()]

    return labels, stacked

def check_data(data):
    if data is None: return None, None
    else:
        # if len(data.shape) == 1: return content_1, np.expand_dims(data, axis=0)
        # if len(data.shape) == 2: return content_2, np.expand_dims(data, axis=0)
        # if len(data.shape) == 3: return content_3, np.expand_dims(data, axis=0)
        # if len(data.shape) == 4: return content_4, np.expand_dims(data, axis=0)
        # if len(data.shape) == 2: return content_1, data
        # if len(data.shape) == 3: return content_2, data
        # if len(data.shape) == 4: return content_3, data
        # if len(data.shape) == 5: return content_4, data
        if len(data.shape) == 1: return content_1, data
        if len(data.shape) == 2: return content_2, data
        if len(data.shape) == 3: return content_3, data
        if len(data.shape) == 4: return content_4, data

def labels(labels, prefix='', suffix='', delimiter='_'):
    if prefix != '': labels = [delimiter.join([prefix, label]) for label in labels]
    if suffix != '': labels = [delimiter.join([label, suffix]) for label in labels]

    return labels

def merge_samples(entities, mode='concat'):
    # join/merge: stack, concat, alternate
    pass

def resolve_samples_as_gen(entity, level='episode', filter=None):
    try:
        for entry in entity.entries:
            if isinstance(entry, LOOKUPS[level]):
                yield from [resolve_samples_as_gen(entry, level)]
            else:
                yield from resolve_samples_as_gen(entry, level)
    except:
        if filter == None:
            yield entity
        elif filter == 'evaluate':
            if entity.static: yield entity
        elif filter == 'train':
            if not entity.static: yield entity

def resolve_samples_as_list(entity, level='episode', filter=None):
    tmp = []
    try:
        for entry in entity.entries:
            if isinstance(entry, LOOKUPS[level]):
                tmp.append(resolve_samples_as_list(entry, level))
            else:
                tmp.extend(resolve_samples_as_list(entry, level))
    except:
        if filter == None:
            tmp.append(entity)
        elif filter == 'evaluate':
            if entity.static: tmp.append(entity)
        elif filter == 'train':
            if not entity.static: tmp.append(entity)
    return tmp

def build_samples(structure, include=[], exclude=[]):
    assert set(include).isdisjoint(exclude) and set(exclude).isdisjoint(include), 'include and exclude are overlapping'

    results = []
    for group in structure:
        data = {}
        for sample in group:
            for attribute, value in vars(sample).items():
                if (include and attribute not in include) or (exclude and attribute in exclude):
                    continue
                try: data[attribute].append(value)
                except: data[attribute] = [value]
        results.append({key: np.array(values) for key, values in data.items()})

    return results

def group_samples(structure, mode=None):
    results = {key: [] for key in structure[0]}
    for group in structure:
        for key, value in group.items():
            if mode == 'append': results[key].append(value)
            if mode == 'extend': results[key].extend(value)

    return results

def aggregate_samples(integrate=None, separate=None):
    pass
