import time
import pprint

import numpy as np
import scipy as sp
import pandas as pd

import seaborn as sns
import matplotlib as mpl

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def generic_init(size=(19.2, 12), dim=2):
    if dim == 2: return plt.subplots(figsize=size)
    elif dim == 3: return plt.subplots(figsize=size, subplot_kw={'projection': '3d'})

def generic_finish(fig, backend, benchmark, tighten=False):
    if tighten: plt.tight_layout()
    plt.savefig(f'./{backend}_{benchmark}.png')
    plt.close(fig)

def benchmark_imshow(axs, data):
    axs.imshow(*data)

def benchmark_hist_1d(axs, data):
    axs.hist(*data)

def benchmark_boxplot(axs, data):
    axs.boxplot(*data)

def benchmark_violine(axs, data):
    axs.violinplot(*data)

def benchmark_curve(axs, data):
    axs.plot(*data)

def benchmark_scatter_2d(axs, data):
    axs.scatter(*data)

def benchmark_hist_2d(axs, data):
    axs.hist2d(*data)

def benchmark_heatmap(axs, data):
    axs.pcolormesh(*data)

def benchmark_contur_2d(axs, data):
    axs.contour(*data)

def benchmark_surface(axs, data):
    axs.plot_surface(*data)

def benchmark_wireframe(axs, data):
    axs.plot_wireframe(*data)

def benchmark_contur_3d(axs, data):
    axs.contour(*data)

def benchmark_scatter_3d(axs, data):
    axs.scatter(*data)


N = 10

IMAGE = [
    np.random.random((5001, 5001, 4))
]

DATA_1 = [
    np.random.random((100000, 30)),
]
DATA_2 = [
    np.random.random((10000, 12)),
]
DATA_3 = [
    np.random.random((50000)),
    np.random.random((50000)),
]
DATA_4 = [
    np.arange(10000),
    np.random.random((10000, 3)),
]
DATA_5 = [
    np.random.random((10000, 12)),
    np.random.random((10000, 12)),
]
DATA_6 = [
    np.linspace(-5, +5, 1000),
    np.linspace(-5, +5, 1000),
]
DATA_6.append((1 - DATA_6[0]/2 + DATA_6[0]**5 + DATA_6[1]**3) * np.exp(-DATA_6[0]**2 - DATA_6[1]**2))
DATA_7 = [
    np.random.random((100, 1000, 2)),
    np.random.random((100, 1000, 2)),
    np.random.random((100, 1000, 2)),
]

BENCHMARKS = [
    (benchmark_imshow, 2, IMAGE),
    (benchmark_hist_1d, 2, DATA_1),
    (benchmark_boxplot, 2, DATA_1),
    (benchmark_violine, 2, DATA_2),
    (benchmark_hist_2d, 2, DATA_3),
    (benchmark_curve, 2, DATA_4),
    (benchmark_scatter_2d, 2, DATA_5),
    (benchmark_heatmap, 2, DATA_6),
    (benchmark_contur_2d, 2, DATA_6),
    # (benchmark_surface, 3, DATA_6),
    # (benchmark_wireframe, 3, DATA_6),
    # (benchmark_contur_3d, 3, DATA_6),
    # (benchmark_scatter_3d, 3, DATA_7),
]

times = {}
for backend in ['Qt5Agg', 'Qt5Cairo']:
    times.update({backend: {}})
    mpl.use(backend, force=True)
    print('backend:', mpl.get_backend())

    for function, dim, data in BENCHMARKS:
        benchmark = function.__name__
        times[backend].update({benchmark: []})
        print('benchmark:', benchmark)

        for i in range(N):
            print(f'{i + 1}/{N}\r', end='')
            start = time.time_ns()
            fig, axs = generic_init((19.2, 12), dim)
            try: function(axs, data)
            except Exception as e: print(e)
            generic_finish(fig, backend, benchmark)
            end = time.time_ns()
            times[backend][benchmark].append(end - start)

        durations = np.divide(times[backend][benchmark], 1e9)
        times[backend][benchmark] = durations

        print('ELAPSED TIME:')
        print(f'avg: {np.round(np.average(durations), 3)} s\tptp: {np.round(np.ptp(durations), 3)} s')
        print(f'min: {np.round(np.min(durations), 3)} s\tmax: {np.round(np.max(durations), 3)} s')
        print(f'std: {np.round(np.std(durations), 3)} s\tvar: {np.round(np.var(durations), 3)} s')

pprint.pprint(times)

df = pd.DataFrame.from_dict(times)
melted_df = df.reset_index().melt(id_vars='index')
exploded_df = melted_df.explode('value')

sns.violinplot(exploded_df, x='value', y='index', hue='variable')

plt.xlabel('Benchmark')
plt.ylabel('Time')
plt.legend()
plt.show()
