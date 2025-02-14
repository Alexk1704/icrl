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

from gazebo_sim.utils.Evaluation import *


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

def finish(kwargs, fig=None, axs=None, flag=''):
    if fig is None: fig = V.current_figure()
    if axs is None: axs = V.current_plot()

    path = None
    if kwargs['mode'] == 'save':
        dirname = os.path.join(kwargs['path'], kwargs['module'], flag)
        filename = '.'.join([kwargs['name'], kwargs['format']])
        path = os.path.join(dirname, filename)
        tools.ensure_path(path)
    V.generate(fig, path)

def generic_plot(data, labels, fig=None, axs=None, flag=''):
    func, data = check_data(data)
    if func is not None and data is not None:
        fig, axs = create(KWARGS, fig, axs)
        func(data, labels, KWARGS, fig, axs, flag)
        finish(KWARGS, fig, axs, flag)

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

def content_3(data, labels, kwargs, fig, axs, flag):
    for i, entry in enumerate(data):
        # if i > 0 and kwargs['plot'] == 'multiple':
        #     fig, axs = create(kwargs, fig, None)

        # entry = A.clip(entry, -2, +1000)
        print('###')
        print('stats:', A.stats(entry))
        print('###')

        counters = DATA['info']['counters']
        sequence = DATA['info']['tasks']

        tasks = []
        for index, task in enumerate(sequence):
            if '' in task['mode']: tasks.append(index)
            # if 'eval' in task['mode']: tasks.append(index)
            # if 'train' in task['mode']: tasks.append(index)

        counters = [counters[task] for task in tasks]
        sequence = [sequence[task] for task in tasks]

        if flag in ['s', 'a', 'r', 'ra', 'raav1', 'raav2']:
            temp = []
            lower, upper = 0, 0
            for episodes in counters:
                for episode in episodes:
                    upper = lower + episode
                    temp.extend(entry[lower:upper])
                    lower = upper

            entry = np.array(temp)

            index = 0
            for t, episodes in enumerate(counters[:-1]):
                axs.text(np.average([index, index + sum(episodes)]), 0, f' T{t + 1}', ha='center', va='bottom', rotation=90)
                # axs.text(np.average([index, index + sum(episodes)]), 0, f' Task {t + 1} ({sequence[t]["mode"]})', ha='center', va='bottom', rotation=90)
                index += sum(episodes)
                # index += len(episodes)
                axs.axvline(index, color='black', alpha=0.75, linewidth=1, linestyle=(0, (5, 10)))
            axs.text(np.average([index, index + sum(counters[-1])]), 0, f' T{t + 2}', ha='center', va='bottom', rotation=90)
            # axs.text(np.average([index, index + sum(counters[-1])]), 0, f' Task {t + 2} ({sequence[t + 1]["mode"]})', ha='center', va='bottom', rotation=90)

            terminal = np.full_like(temp, np.nan)

            index = 0
            for episodes in counters:
                for episode in episodes[:-1]:
                    index += episode
                    terminal[index] = 0
                index += episodes[-1]

        if flag == 's':
            # smoothed state
            temp = pd.Series(np.reshape(entry, -1)).rolling(window=10).mean()
            # temp = pd.Series(np.reshape(entry, -1)).rolling(window=10).median()

            #fig, axs = create(kwargs, fig, None)
            V.curve_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5)
            # V.scatter_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5, s=10)
            V.scatter_2D(axs, labels, None, terminal, alpha=0.75, linewidth=0.5, s=10, c='red', marker='x')

            V.current_plot().set_title('observed states')
            V.current_plot().set_xlabel('agent step')
            V.current_plot().set_ylabel('normalized value')

        if flag == 'a':
            # smoothed action
            temp = np.subtract(entry[:, 0], entry[:, 1])

            temp = pd.Series(np.reshape(temp, -1)).rolling(window=10).mean()
            # temp = pd.Series(np.reshape(temp, -1)).rolling(window=10).median()

            #fig, axs = create(kwargs, fig, None)
            V.curve_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5)
            # V.scatter_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5, s=10)
            V.scatter_2D(axs, labels, None, terminal, alpha=0.75, linewidth=0.5, s=10, c='red', marker='x')

            V.current_plot().set_title('conducted action')
            V.current_plot().set_xlabel('agent step')
            V.current_plot().set_ylabel('direction tendency')

        if flag == 'ac':
            # accumulated action
            temp = np.subtract(entry[:, 0], entry[:, 1])

            temp = pd.Series(np.reshape(temp, -1)).cumsum()

            #fig, axs = create(kwargs, fig, None)
            V.curve_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5)
            # V.scatter_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5, s=10)
            V.scatter_2D(axs, labels, None, terminal, alpha=0.75, linewidth=0.5, s=10, c='red', marker='x')

            V.current_plot().set_title('conducted action')
            V.current_plot().set_xlabel('agent step')
            V.current_plot().set_ylabel('accumulated direction')

        if flag == 'accv1':
            # averaged accumulated action (sum-avg)
            temp = np.subtract(entry[:, 0], entry[:, 1])

            temp = pd.Series(np.reshape(temp, -1)).cumsum()
            temp = pd.Series(np.reshape(temp, -1)).rolling(window=10).mean()
            # temp = pd.Series(np.reshape(temp, -1)).rolling(window=10).median()

            #fig, axs = create(kwargs, fig, None)
            V.curve_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5)
            # V.scatter_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5, s=10)
            V.scatter_2D(axs, labels, None, terminal, alpha=0.75, linewidth=0.5, s=10, c='red', marker='x')

            V.current_plot().set_title('conducted action')
            V.current_plot().set_xlabel('agent step')
            V.current_plot().set_ylabel('accumulated direction')

        if flag == 'accv2':
            # averaged accumulated action (avg-sum)
            temp = np.subtract(entry[:, 0], entry[:, 1])

            temp = pd.Series(np.reshape(temp, -1)).rolling(window=10).mean()
            # temp = pd.Series(np.reshape(temp, -1)).rolling(window=10).median()
            temp = pd.Series(np.reshape(temp, -1)).cumsum()

            #fig, axs = create(kwargs, fig, None)
            V.curve_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5)
            # V.scatter_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5, s=10)
            V.scatter_2D(axs, labels, None, terminal, alpha=0.75, linewidth=0.5, s=10, c='red', marker='x')

            V.current_plot().set_title('conducted action')
            V.current_plot().set_xlabel('agent step')
            V.current_plot().set_ylabel('accumulated direction')

        if flag == 'r':
            # averaged reward
            temp = pd.Series(np.reshape(entry, -1)).rolling(window=10).mean()
            # temp = pd.Series(np.reshape(entry, -1)).rolling(window=10).median()

            #fig, axs = create(kwargs, fig, None)
            V.curve_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5)
            # V.scatter_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5, s=10)
            V.scatter_2D(axs, labels, None, terminal, alpha=0.75, linewidth=0.5, s=10, c='red', marker='x')

            V.current_plot().set_title('reward signal')
            V.current_plot().set_xlabel('agent step')
            V.current_plot().set_ylabel('normalized value')

        if flag == 'ra':
            # accumulated reward
            temp = pd.Series(np.reshape(entry, -1)).cumsum()

            #fig, axs = create(kwargs, fig, None)
            V.curve_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5)
            # V.scatter_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5, s=10)
            V.scatter_2D(axs, labels, None, terminal, alpha=0.75, linewidth=0.5, s=10, c='red', marker='x')

            V.current_plot().set_title('reward signal')
            V.current_plot().set_xlabel('agent step')
            V.current_plot().set_ylabel('accumulated value')

        if flag == 'raav1':
            # averaged accumulated reward (sum-avg)
            temp = pd.Series(np.reshape(entry, -1)).cumsum()
            temp = pd.Series(np.reshape(temp, -1)).rolling(window=10).mean()
            # temp = pd.Series(np.reshape(temp, -1)).rolling(window=10).median()

            #fig, axs = create(kwargs, fig, None)
            V.curve_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5)
            # V.scatter_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5, s=10)
            V.scatter_2D(axs, labels, None, terminal, alpha=0.75, linewidth=0.5, s=10, c='red', marker='x')

            V.current_plot().set_title('reward signal')
            V.current_plot().set_xlabel('agent step')
            V.current_plot().set_ylabel('accumulated value')

        if flag == 'raav2':
            # averaged accumulated reward (avg-sum)
            temp = pd.Series(np.reshape(entry, -1)).rolling(window=10).mean()
            # temp = pd.Series(np.reshape(entry, -1)).rolling(window=10).median()
            temp = pd.Series(np.reshape(temp, -1)).cumsum()

            #fig, axs = create(kwargs, fig, None)
            V.curve_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5)
            # V.scatter_2D(axs, labels, None, temp, alpha=0.75, linewidth=0.5, s=10)
            V.scatter_2D(axs, labels, None, terminal, alpha=0.75, linewidth=0.5, s=10, c='red', marker='x')

            V.current_plot().set_title('reward signal')
            V.current_plot().set_xlabel('agent step')
            V.current_plot().set_ylabel('accumulated value')

        if flag in ['rs', 'rl', 're']:
            lower, upper = 0, 0
            score, length = [], []
            for episodes in counters:
                for episode in episodes:
                    upper = lower + episode
                    score.append(sum(entry[lower:upper]))
                    length.append(len(entry[lower:upper]))
                    lower = upper

            score = np.array(score).flatten()
            length = np.array(length).flatten()
            effect = np.divide(score, length)

            # print(score.shape)
            # print(length.shape)
            # print(effect.shape)

            index = 0
            for t, episodes in enumerate(counters[:-1]):
                axs.text(np.average([index, index + len(episodes)]), 0, f' T{t + 1}', ha='center', va='bottom', rotation=90)
                # axs.text(np.average([index, index + len(episodes)]), 0, f' Task {t + 1} ({sequence[t]["mode"]})', ha='center', va='bottom', rotation=90)
                # index += sum(episodes)
                index += len(episodes)
                axs.axvline(index, color='black', alpha=0.75, linewidth=1, linestyle=(0, (5, 10)))
            axs.text(np.average([index, index + len(counters[-1])]), 0, f' T{t + 2}', ha='center', va='bottom', rotation=90)
            # axs.text(np.average([index, index + len(counters[-1])]), 0, f' Task {t + 2} ({sequence[t + 1]["mode"]})', ha='center', va='bottom', rotation=90)

        if flag == 'rs':
            # scatter by score and length
            #fig, axs = create(kwargs, fig, None)
            V.scatter_2D(axs, labels, None, score, c=score, s=length, alpha=0.75, linewidth=0.5, cmap='viridis')
            # V.scatter_2D(axs, labels, None, score, c=length, s=length, alpha=0.75, linewidth=0.5, cmap='viridis')

            V.current_plot().set_title('trajectory by score')
            V.current_plot().set_xlabel('agent episode')
            V.current_plot().set_ylabel('total value')

        if flag == 'rl':
            # scatter by length and score
            #fig, axs = create(kwargs, fig, None)
            V.scatter_2D(axs, labels, None, length, c=length, s=score, alpha=0.75, linewidth=0.5, cmap='viridis')
            # V.scatter_2D(axs, labels, None, length, c=score, s=score, alpha=0.75, linewidth=0.5, cmap='viridis')

            V.current_plot().set_title('trajectory by length')
            V.current_plot().set_xlabel('agent episode')
            V.current_plot().set_ylabel('total value')

        if flag == 're':
            # scatter by length and score
            #fig, axs = create(kwargs, fig, None)
            V.scatter_2D(axs, labels, None, effect, c=score, s=length, alpha=0.75, linewidth=0.5, cmap='viridis')
            # V.scatter_2D(axs, labels, None, effect, c=length, s=score, alpha=0.75, linewidth=0.5, cmap='viridis')

            V.current_plot().set_title('trajectory by effect (score / length)')
            V.current_plot().set_xlabel('agent episode')
            V.current_plot().set_ylabel('total value')

        if flag == 're':
            result = {}

            lookup_array = {
                'raw': entry
            }

            for key, value in lookup_array.items():
                result[key] = []

                lower, upper = 0, 0
                for episodes in counters:
                    upper = lower + sum(episodes)
                    # upper = lower + len(episodes)
                    task = value[lower:upper]
                    lower = upper

                    result[key].append({
                        'min': float(np.min(task, axis=None)),
                        'max': float(np.max(task, axis=None)),
                        'ptp': float(np.ptp(task, axis=None)),
                        'avg': float(np.average(task, axis=None)),
                        'var': float(np.var(task, axis=None)),
                        'std': float(np.std(task, axis=None)),
                    })

            lookup_array = {
                'score': score,
                'length': length,
                'effect': effect,
            }

            for key, value in lookup_array.items():
                result[key] = []

                lower, upper = 0, 0
                for episodes in counters:
                    # upper = lower + sum(episodes)
                    upper = lower + len(episodes)
                    task = value[lower:upper]
                    lower = upper

                    result[key].append({
                        'min': float(np.min(task, axis=None)),
                        'max': float(np.max(task, axis=None)),
                        'ptp': float(np.ptp(task, axis=None)),
                        'avg': float(np.average(task, axis=None)),
                        'var': float(np.var(task, axis=None)),
                        'std': float(np.std(task, axis=None)),
                    })

            print('###')
            print('results:', result)
            print('###')

            if kwargs['mode'] == 'save':
                dirname = os.path.join(kwargs['path'], kwargs['module'])
                filename = '.'.join([kwargs['name'], 'json'])
                path = os.path.join(dirname, filename)
                tools.ensure_path(path)

                import json
                with open(path, 'w') as fp:
                    json.dump(result, fp)


        if flag in ['sd', 'ad', 'rd']:
            shape = entry.shape[1:]
            entry = entry[~np.isnan(entry)]
            entry = entry.reshape((-1, *shape))

            # fig, axs = create(kwargs, fig, None)
            # labels, entry = np.unique(entry, return_counts=True)
            # V.pie_1D(axs, labels, entry)

            # fig, axs = create(kwargs, fig, None)
            if 'a' in flag:
                uniq = np.unique(entry)
                bins = [*uniq, uniq[-1] + (uniq[-1] - uniq[-2])]
                V.hist_1D(axs, labels, entry.astype(np.float32), bins=bins, align='left')
            else:
                V.hist_1D(axs, labels, entry.astype(np.float32))

            if 's' in flag:
                V.current_plot().set_title('state frequency')
                V.current_plot().set_xlabel('value')
                V.current_plot().set_ylabel('count')
            if 'a' in flag:
                V.current_plot().set_title('action frequency')
                V.current_plot().set_xlabel('value')
                V.current_plot().set_ylabel('count')
            if 'r' in flag:
                V.current_plot().set_title('action frequency')
                V.current_plot().set_xlabel('value')
                V.current_plot().set_ylabel('count')

            fig, axs = create(kwargs, fig, None)
            V.violin_1D(axs, labels, entry.astype(np.float32), orientation='h')

            if 's' in flag:
                V.current_plot().set_title('state distribution')
                V.current_plot().set_xlabel('value')
                V.current_plot().set_ylabel('frequency')
            if 'a' in flag:
                V.current_plot().set_title('action distribution')
                V.current_plot().set_xlabel('value')
                V.current_plot().set_ylabel('frequency')
            if 'r' in flag:
                V.current_plot().set_title('action distribution')
                V.current_plot().set_xlabel('value')
                V.current_plot().set_ylabel('frequency')

        '''
        fig, axs = create(kwargs, fig, None)
        V.curve_2D(axs, labels, None, entry, alpha=0.75, linewidth=0.5)
        fig, axs = create(kwargs, fig, None)
        V.scatter_2D(axs, labels, None, entry, alpha=0.75, linewidth=0.5, s=10)
        '''

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
        # stacked = [{key: np.concatenate([entry[key].reshape(-1, 1, entry[key].shape[-1]) for entry in value]) for key in value[0].keys()} for value in wrapper.values()]
        stacked = [{key: [entry[key].reshape(-1, 1, entry[key].shape[-1]) for entry in value] for key in value[0].keys()} for value in wrapper.values()]

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
