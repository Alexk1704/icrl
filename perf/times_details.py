import os
import re
import sys
import glob
import uuid
import morph
import pprint
import argparse
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import multiprocessing as mp


def read_files(path, pattern):
    for file in sorted(glob.glob(os.path.join(path, pattern), recursive=True)):
        if file.find('#') == -1:
            with open(file, 'r') as fp:
                raw = fp.read()
            file_name = os.path.splitext(os.path.relpath(file, path))[0]

            data = {}
            matches = re.findall(r'([\w\ ]+) took \[\[([\d\.]+)\] ms\]', raw)
            for match in matches:
                name, time = match
                # if name.find('counter') != -1:
                try: data[name.strip()].append(float(time))
                except: data[name.strip()] = [float(time)]

            yield file_name, pd.DataFrame().from_dict(data, orient='index').transpose()

            # cnt = 0
            # data = []
            # depth = 0
            # lines = raw.splitlines()
            # while cnt < len(lines):
            #     try:
            #         context, task = re.findall(r'Switch training (\w+) on track (\w+)', lines[cnt])[0]
            #         if context == 'True': context = 'train'
            #         if context == 'False': context = 'eval'
            #         data.append([])
            #     except: pass

            #     try:
            #         match = re.findall(r'([\w\ ]+) took \[\[([\d\.]+)\] ms\]', lines[cnt])[0]
            #         name, time = match

            #         try:
            #             last_time = re.findall(r'([\w\ ]+) took \[\[([\d\.]+)\] ms\]', lines[cnt-1])[0][1]
            #             depth += 1
            #         except:
            #             last_time = '0.0'
            #             depth = 0

            #         key = f'{context} {task} {name.strip()} (depth={depth})'
            #         key = f'{context} {name.strip()} (depth={depth})'
            #         # value = float(time)
            #         value = float(time) - float(last_time)

            #         data[-1].append({key: value})
            #     except: pass
            #     cnt += 1

            # yield file_name, [pd.DataFrame(entry) for entry in data]


def filter_data(struct, includes=[], excludes=[]):
    filtered = struct
    if len(includes) > 0:
        filtered = {
            key: value for key, value in filtered.items()
            if any([key.find(x) != -1 for x in includes])
        }
    if len(excludes) > 0:
        filtered = {
            key: value for key, value in filtered.items()
            if all([key.find(x) == -1 for x in excludes])
        }

    return morph.unflatten({'.'.join(key.rsplit('/', 1)): value for key, value in filtered.items()})


def init_plot(number, titel='', share=('none', 'none')):
    grid = np.sqrt(number)
    rows = int(np.round(grid)) # alignment r-c or c-r
    cols = int(np.ceil(grid))  # alignment r-c or c-r

    sns.set_theme()
    fig, axes = plt.subplots(rows, cols, sharex=share[0], sharey=share[1], figsize=(19.2, 12))
    axes = np.reshape(axes, (rows, cols))
    if titel != '': fig.suptitle(titel)

    return axes


def show_plot(show=True, store=False):
    plt.tight_layout()
    if show:
        plt.show()
    if store:
        fig_id = str(uuid.uuid4())
        print('saving:', os.path.join(cwd, 'figs/backend', fig_id))
        plt.savefig(os.path.join(cwd, 'figs/backend', fig_id))


def plot_times(data, levels=[], plot='', overlap=False, threshold=None):
    def fill_plot(frames, axis, title='', plot='', overlap=False, threshold=None):
        entry = pd.concat(frames)
        # entry = pd.concat(frames[0], ignore_index=(not overlap))
        # entry = entry.loc[:, [col.find('train') != -1 for col in entry.columns]]

        if threshold != None:
            const = entry[entry.index < threshold]
            scale = entry[entry.index >= threshold]

        if title != '': axis.set_title(title)

        # entry = entry[entry < 2500]
        # entry = entry.loc[:, 'train reset (depth=1)']

        if plot == 'scatter':
            axis.set_xlabel('episode')
            axis.set_ylabel('time in ms')

            try:
                sns.scatterplot(ax=axis, data=const)
                sns.scatterplot(ax=axis, data=scale)
            except:
                sns.scatterplot(ax=axis, data=entry)

        if plot == 'boxplot':
            axis.set_xlabel('time in ms')

            try:
                sns.boxplot(ax=axis, orient='y', data=const)
                sns.boxplot(ax=axis, orient='y', data=scale)
            except:
                sns.boxplot(ax=axis, orient='y', data=entry)

    axes = init_plot(len(data), 'Performance measures')

    for cnt, (key, value) in enumerate(data.items()):
        row = cnt // axes.shape[0]
        col = cnt % axes.shape[1]
        if axes.shape[0] == 1: col = 0
        if axes.shape[1] == 1: row = 0
        fill_plot(list(value.values()), axes[col, row], key, plot, overlap, threshold)

    show_plot(show=False, store=True)


try: merge = bool(int(sys.argv[1]))
except: merge = True
try: limit = int(sys.argv[2])
except: limit = None

cwd = os.path.dirname(os.path.realpath(__file__))

def mt_func(pattern):
    print('processing:', pattern)

    wrapper = dict(read_files(cwd, pattern))
    data = filter_data(wrapper, includes=['BACKEND'])

    try:
        plot_times(data, plot='scatter', overlap=merge, threshold=limit)
        # plot_times(data, plot='boxplot', overlap=merge, threshold=limit)
    except: pass

# for pattern in sorted(glob.glob(os.path.join(cwd, '**/*.log'), recursive=True)):
#     mt_func(pattern)

with mp.Pool() as pool:
    pool.map(mt_func, sorted(glob.glob(os.path.join(cwd, '**/*.log'), recursive=True)))
