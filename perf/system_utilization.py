import os
import sys
import glob
import morph
import pickle
import pprint
import typing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import multiprocessing as mp
import matplotlib.pyplot as plt


def find_files(path, pattern):
    return sorted(glob.glob(os.path.join(path, pattern), recursive=True))


def ensure_path(path):
    if os.path.exists(path):
        return True
    else:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        return False


def load_data(file):
    with open(file, 'rb') as fp:
        data = pickle.load(fp)
    return data


def convert_data(data):
    def resolve(entry):
        if isinstance(entry, dict):
            for key, value in entry.items():
                try: entry[key] = resolve(value._asdict())
                except: entry[key] = resolve(value)
        elif isinstance(entry, list):
            for i, element in enumerate(entry):
                try: entry[i] = resolve(element._asdict())
                except: entry[i] = resolve(element)
        elif isinstance(entry, tuple):
            entry = list(entry)
            for i, element in enumerate(entry):
                try: entry[i] = resolve(element._asdict())
                except: entry[i] = resolve(element)
        return entry
    return resolve(data)


def create_dataframe(data):
    return pd.DataFrame(data)


def concat_dataframe(data, ignore_index=True):
    return pd.concat(data, ignore_index=ignore_index)


def resolve_filters(iterable):
    includes, excludes = [], []
    for entry in iterable:
        if entry.startswith('+'): includes.append(entry.removeprefix('+'))
        if entry.startswith('-'): excludes.append(entry.removeprefix('-'))
    return includes, excludes


def filter_iterable(iterable, includes=[], excludes=[]):
    iterable = list(iterable)
    for entry in iterable.copy():
        if len(includes) > 0:
            if all(entry.find(x) == -1 for x in includes):
                iterable.remove(entry)
        if len(excludes) > 0:
            if any(entry.find(x) != -1 for x in excludes):
                iterable.remove(entry)
    return iterable


def filter_dataframe(df, includes=[], excludes=[]):
    include_mask = df.columns.isin(includes)
    exclude_mask = ~df.columns.isin(excludes)

    df_filtered = df.loc[:, include_mask | exclude_mask]
    return df_filtered


def create_plot(wrapper, xscale='linear', xshare=False, yscale='linear', yshare=False, output_dir='', file_name=[]):
    sns.set_theme(rc={'figure.figsize': (19.2, 12)})

    number = int(np.ceil(np.sqrt(len(wrapper))))
    g = sns.FacetGrid(pd.concat(wrapper), col='name', col_wrap=number, sharex=xshare, sharey=yshare)
    g.figure.suptitle(f'{" ".join(file_name)} utilization')
    # g.figure.autofmt_xdate()

    g.set(xscale=xscale)
    g.set(yscale=yscale)
    g.set(xlabel='timestamp')
    g.set(ylabel='utilization')

    g.map(sns.lineplot, 'timestamp', 'value', 'variable', alpha=0.9, linewidth=0.25)
    g.add_legend()

    for ax in g.axes.flat:
        ax.xaxis.set_tick_params(rotation=30)

    plt.tight_layout()

    if output_dir == '':
        plt.show()
    else:
        ensure_path(output_dir)
        plt.savefig(os.path.join(output_dir, '_'.join(file_name)))


def resolve_labels(df):
    labels = [{entry['label'] for entry in df[c]} for c in df.columns]
    return [f'{" ".join(c)} ({i})' for i, c in enumerate(labels)]


def melt_plot(org_df, alt_df, name, columns=[]):
    alt_df['timestamp'] = org_df['timestamp']
    alt_df = filter_dataframe(alt_df, *resolve_filters(columns))

    # plot each ten-th entry
    # alt_df = alt_df.iloc[::100]
    # plot only 1000 entries
    # alt_df.sample(n=1000)

    alt_df = alt_df.groupby(alt_df.index // 100).aggregate('mean')

    # alt_df = alt_df.groupby(alt_df.index // 100).aggregate('mean')
    # alt_df = alt_df.groupby(alt_df.index // 100).transform('mean')
    # alt_df = alt_df.rolling(window=100).mean()

    df_melted = alt_df.melt(id_vars='timestamp')
    df_melted['name'] = name

    return df_melted


def plot_first_level(data, name, columns=[], filters=[], separated=False, output_dir=''):
    wrapper = []
    df = pd.DataFrame.from_records(data[name])

    if not separated:
        wrapper.append(melt_plot(data, df, name, columns))
        create_plot(wrapper, output_dir=output_dir, file_name=[name])
    else:
        for col in filter_iterable(df.columns, *resolve_filters(filters)):
            _df = pd.concat([pd.DataFrame([entry]) for entry in df[col]], ignore_index=True)
            wrapper.append(melt_plot(data, _df, col, columns))
        create_plot(wrapper, output_dir=output_dir, file_name=[name])


def plot_second_level(data, name, columns=[], filters=[], separated=False, output_dir=''):
    wrapper = []
    df = pd.DataFrame.from_records(data[name])

    if not separated:
        for col in filter_iterable(df.columns, *resolve_filters(filters)):
            _df = pd.DataFrame.from_records(df[col])
            wrapper.append(melt_plot(data, _df, col, columns))
        create_plot(wrapper, output_dir=output_dir, file_name=[name])
    else:
        for col in filter_iterable(df.columns, *resolve_filters(filters)):
            wrapper.clear()
            _df = pd.DataFrame.from_records(df[col])
            for _col in filter_iterable(_df.columns, *resolve_filters(filters)):
                __df = pd.concat([pd.DataFrame([entry]) for entry in _df[_col]], ignore_index=True)
                wrapper.append(melt_plot(data, __df, _col, columns))
            create_plot(wrapper, output_dir=output_dir, file_name=[name, col])


def plot_third_level(data, name, columns=[], filters=[], separated=False, output_dir=''):
    wrapper = []
    df = pd.DataFrame.from_records(data[name])

    if not separated:
        for col in filter_iterable(df.columns, *resolve_filters(filters)):
            wrapper.clear()
            _df = pd.DataFrame.from_records(df[col])
            _df.columns = resolve_labels(_df)
            for _col in filter_iterable(_df.columns, *resolve_filters(filters)):
                __df = pd.DataFrame.from_records(_df[_col])
                __df = filter_dataframe(__df, excludes=['label'])
                wrapper.append(melt_plot(data, __df, _col, columns))
            create_plot(wrapper, output_dir=output_dir, file_name=[name, col])
    else:
        for col in filter_iterable(df.columns, *resolve_filters(filters)):
            _df = pd.DataFrame.from_records(df[col])
            _df.columns = resolve_labels(_df)
            for _col in filter_iterable(_df.columns, *resolve_filters(filters)):
                wrapper.clear()
                __df = pd.DataFrame.from_records(_df[_col])
                __df = filter_dataframe(__df, excludes=['label'])
                for __col in filter_iterable(__df.columns, *resolve_filters(filters)):
                    ___df = pd.concat([pd.DataFrame([entry]) for entry in __df[__col]], ignore_index=True)
                    wrapper.append(melt_plot(data, ___df, __col, columns))
                create_plot(wrapper, output_dir=output_dir, file_name=[name, col, _col])


cwd = os.path.dirname(os.path.realpath(__file__))
# lookup_path = os.path.expanduser('~/Desktop/Experiments/v4/CRL_TEST_DQN__24-03-21-20-53-15/results/success/CRL_TEST_DQN__E0-W0-C0-R0')
# lookup_path = os.path.expanduser('~/Desktop/Experiments/v4/CRL_TEST_QGMM__24-03-21-20-52-57/results/success/CRL_TEST_QGMM__E0-W0-C0-R0')
# lookup_path = os.path.expanduser('~/Desktop/Experiments/v5/QGMM_ABORTED_1')
lookup_path = os.path.expanduser('~/Downloads/profiler')

separated = True
output_dir = ''


def mt_wrapper(file):
    print(file)
    data = load_data(file)
    data = convert_data(data)
    data = create_dataframe(data)
    return data

with mp.Pool() as pool:
    print('search files')
    files = find_files(lookup_path, '**/stats/Profiler/*.pkl')
    files = filter_iterable(files)

    print('process files')
    wrapper = pool.map(mt_wrapper, files)

    print('build dataframe')
    data = concat_dataframe(wrapper)

    print('generate plots')
    if True:
        plot_first_level(data, 'cpu', separated=separated, output_dir=output_dir)
        plot_first_level(data, 'times', separated=separated, output_dir=output_dir)
        plot_first_level(data, 'mem', separated=separated, output_dir=output_dir, columns=['-total'])
        plot_first_level(data, 'swap', separated=separated, output_dir=output_dir)
        # plot_first_level(data, 'gpu', separated=separated, output_dir=output_dir)
        plot_second_level(data, 'blk', separated=separated, output_dir=output_dir, filters=['-loop'])
        plot_second_level(data, 'net', separated=separated, output_dir=output_dir)
        plot_third_level(data, 'temps', separated=separated, output_dir=output_dir)
        plot_third_level(data, 'fans', separated=separated, output_dir=output_dir)
    else:
        results = []
        results.append(pool.apply_async(plot_first_level, args=[data, 'cpu'], kwds={'separated': separated, 'output_dir': output_dir}))
        results.append(pool.apply_async(plot_first_level, args=[data, 'times'], kwds={'separated': separated, 'output_dir': output_dir}))
        results.append(pool.apply_async(plot_first_level, args=[data, 'mem'], kwds={'separated': separated, 'output_dir': output_dir, 'columns': ['-total']}))
        results.append(pool.apply_async(plot_first_level, args=[data, 'swap'], kwds={'separated': separated, 'output_dir': output_dir}))
        # results.append(pool.apply_async(plot_first_level, args=[data, 'gpu'], kwds={'separated': separated, 'output_dir': output_dir}))
        results.append(pool.apply_async(plot_second_level, args=[data, 'blk'], kwds={'separated': separated, 'output_dir': output_dir, 'filters': ['-loop']}))
        results.append(pool.apply_async(plot_second_level, args=[data, 'net'], kwds={'separated': separated, 'output_dir': output_dir}))
        results.append(pool.apply_async(plot_third_level, args=[data, 'temps'], kwds={'separated': separated, 'output_dir': output_dir}))
        results.append(pool.apply_async(plot_third_level, args=[data, 'fans'], kwds={'separated': separated, 'output_dir': output_dir}))
        [result.get() for result in results]
