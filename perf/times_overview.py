import os
import re
import sys
import glob
import morph
import pprint
import argparse
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_files(path, pattern):
    for file in sorted(glob.glob(os.path.join(path, pattern), recursive=True)):
        if file.find('#') == -1:
            with open(file, 'r') as fp:
                raw = fp.read()
            file_name = os.path.splitext(os.path.relpath(file, path))[0]

            data = {}
            # matches = re.findall(r'>>> (.*)\s*?@\s*?(\d+\.\d{6})', raw)
            matches = re.findall(r'>>> ([\w\ ]+)\s*?@\s*?(\d+\.\d{6})', raw)
            for match in matches:
                name, time = match
                # if name.find('counter') != -1:
                try: data[name.strip()].append(float(time))
                except: data[name.strip()] = [float(time)]

            yield file_name, pd.DataFrame().from_dict(data, orient='index').transpose()


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


def show_plot():
    plt.tight_layout()
    plt.show()


def plot_times(data, machines, levels=[], aggregate='', threshold=0):
    def fill_plot(frames, axis, title='', aggregate='', threshold=0):
        df = pd.concat(frames)
        data = df.loc[:, df.aggregate(aggregate) > threshold]
        order = np.flip(data.aggregate(aggregate).sort_values().index)

        if title != '': axis.set_title(title)
        axis.set_xlabel('times in s')
        axis.set_xscale('log')

        sns.boxplot(ax=axis, orient='y', data=data, order=order)

    for machine in machines:
        data = {key: value for key, value in data.items() if key.find(machine) != -1}

        axes = init_plot(len(data), 'Performance measures', ('all', 'row'))

        for cnt, (key, value) in enumerate(data.items()):
            row = cnt // axes.shape[0]
            col = cnt % axes.shape[1]
            if axes.shape[0] == 1: col = 0
            if axes.shape[1] == 1: row = 0
            fill_plot(list(value.values()), axes[col, row], key, aggregate, threshold)

        show_plot()


def compare_times(data, machines, aggregate='', threshold=0):
    def fill_plot(frame, axis, title='', aggregate='', threshold=0):
        slower = np.divide(frame['/'.join([machine_1, mode, component])], frame['/'.join([machine_2, mode, component])])
        faster = np.divide(frame['/'.join([machine_2, mode, component])], frame['/'.join([machine_1, mode, component])])
        factors = pd.concat([np.negative(slower[slower > 1]), np.positive(faster[faster > 1])])
        order = factors.sort_values().index

        print(f'slower: {len(slower[slower > 1])}/{len(slower)}')
        print(f'faster: {len(faster[faster > 1])}/{len(faster)}')

        if title != '': axis.set_title(title)
        axis.set_xlabel('times (slower/faster)')
        axis.set_xscale('linear')

        sns.barplot(ax=axis, orient='y', data=factors, order=order, palette='RdYlGn')

    modes = set()
    components = set()
    for key in data.keys():
        _, mode, component = key.split('/')
        modes.add(mode)
        components.add(component)

    for key, value in data.items():
        df = pd.concat(list(value.values()))
        agg = df.aggregate(aggregate)
        data[key] = agg[agg > threshold]

    for machine_1, machine_2 in itertools.combinations(machines, 2):
        axes = init_plot(np.prod([len(modes), len(components)]), f'Comparison {machine_1} vs {machine_2}')

        for i, mode in enumerate(modes):
            for j, component in enumerate(components):
                fill_plot(data, axes[i, j], f'{mode} - {component}', aggregate, threshold)
                # fill_plot(data, axes[j, i], f'{mode} - {component}', aggregate, threshold)

        show_plot()


try: agg = sys.argv[1]
except: agg = 'mean'
try: val = float(sys.argv[2])
except: val = 0


cwd = os.path.dirname(os.path.realpath(__file__))
wrapper = dict(read_files(cwd, 'logs/**/*.log'))
data = filter_data(wrapper, includes=['backend'])

machines = {key.split('/')[0] for key in data.keys()}
# plot_times(data, machines, aggregate=agg, threshold=val)
compare_times(data, machines, aggregate=agg, threshold=val)
