import os
import sys
import glob

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt


# sns.set_palette('viridis', 24)
sns.set_style('darkgrid')

# groups = glob.glob('/mnt/STORAGE/TEMP/ICRL/*/')
# groups = glob.glob('/mnt/STORAGE/TEMP/ICRL/*24-06-25*/')
# groups = glob.glob('/mnt/STORAGE/TEMP/ICRL/*24-06-26*/')
groups = glob.glob('/mnt/STORAGE/TEMP/ICRL/*24-06-27*/')

experiments = glob.glob('/mnt/STORAGE/TEMP/ICRL/**/evaluation/samples/*.json')

tasks = np.array([3, 2, 1])
reverse = True

prop = 'raw'
func = 'avg'

'''
for group in ['3t', 'base']:
    data = {}
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(19.2, 10.8))
    for experiment in sorted(experiments):
        if experiment.find(group) != -1:
            df = pd.DataFrame.from_records(pd.read_json(experiment)[prop])[func]
            if reverse: df = df[len(df) - tasks]
            else: df = df[0 + tasks]
            name = os.path.basename(experiment)
            data.update({name: df})

            if 'dqn' in experiment:
                # c = 'red'
                c = sns.color_palette('flare', 24)[np.random.randint(24)]
            if 'qgmm' in experiment:
                # c = 'green'
                c = sns.color_palette('mako', 24)[np.random.randint(24)]

            sns.lineplot(df, alpha=0.75, label=name, ax=axs[0], c=c)
            # sns.scatterplot(df, alpha=0.75, label=name, ax=axs[0], c=c)
            sns.boxplot(x=len(data), y=df, label=name, ax=axs[1], color=c)
            # sns.violinplot(x=len(data), y=df, label=name, ax=axs[1], color=c)

    axs[0].legend('')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)

    plt.suptitle(group)
    plt.tight_layout()
    plt.show()
'''

'''
for group in ['3t', 'base']:
    data = {}

    dqn = []
    qgmm = []
    dqn_names = []
    qgmm_names = []
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(19.2, 10.8))
    for experiment in sorted(experiments):
        if experiment.find(group) != -1:
            df = pd.DataFrame.from_records(pd.read_json(experiment)[prop])[func]
            if reverse: df = df[len(df) - tasks]
            else: df = df[0 + tasks]
            name = os.path.basename(experiment).split('.')[0]
            data.update({name: df})

            if 'dqn' in experiment:
                dqn.append(df)
                dqn_names.append(name)
            if 'qgmm' in experiment:
                qgmm.append(df)
                qgmm_names.append(name)

    sums = [df.sum() for df in dqn]
    sorted_dfs = np.argsort(sums)[::-1]

    # c = sns.color_palette('flare')[np.random.randint(len(sns.color_palette('flare')))]
    for i, index in enumerate(sorted_dfs[:10]):
        c = sns.color_palette('flare', 24)[np.random.randint(24)]

        sns.lineplot(dqn[index], alpha=0.75, label=f'{i + 1}. {dqn_names[index]}', ax=axs[0], c=c)
        # sns.scatterplot(dqn[index], alpha=0.75, label=f'{i + 1}. {dqn_names[index]}', ax=axs[0], c=c)
        sns.boxplot(x=i, y=dqn[index], label=f'{i + 1}. {dqn_names[index]}', ax=axs[1], color=c)
        # sns.violinplot(x=i, y=dqn[index], label=f'{i + 1}. {dqn_names[index]}', ax=axs[1], color=c)

    sums = [df.sum() for df in qgmm]
    sorted_dfs = np.argsort(sums)[::-1]

    # c = sns.color_palette('mako')[np.random.randint(len(sns.color_palette('mako')))]
    for i, index in enumerate(sorted_dfs[:10]):
        c = sns.color_palette('mako', 24)[np.random.randint(24)]

        sns.lineplot(qgmm[index], alpha=0.75, label=f'{i + 1}. {qgmm_names[index]}', ax=axs[0], c=c)
        # sns.scatterplot(qgmm[index], alpha=0.75, label=f'{i + 1}. {qgmm_names[index]}', ax=axs[0], c=c)
        sns.boxplot(x=i + 10, y=qgmm[index], label=f'{i + 1}. {qgmm_names[index]}', ax=axs[1], color=c)
        # sns.violinplot(x=i + 10, y=qgmm[index], label=f'{i + 1}. {qgmm_names[index]}', ax=axs[1], color=c)

    axs[0].legend('')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)

    plt.suptitle(group)
    plt.tight_layout()
    plt.show()
'''

'''
for group in sorted(groups):
    data = {}
    for experiment in sorted(experiments):
        if experiment.find(group) != -1:
            df = pd.DataFrame.from_records(pd.read_json(experiment)[prop])[func]
            if reverse: df = df[len(df) - tasks]
            else: df = df[0 + tasks]
            name = os.path.basename(experiment)
            data.update({name: df})

            sns.lineplot(df, alpha=0.75, label=name)
            # sns.scatterplot(df, alpha=0.75, label=name)
            # sns.boxplot(x=len(data), y=df, label=name)
            # sns.violinplot(x=len(data), y=df, label=name)

    plt.title(os.path.basename(os.path.dirname(group)))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
'''

'''
# accumulieren über runs, dann auch weniger daten
for group in ['3t', 'base']:
    data = {}
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(19.2, 10.8))

    filtered_group = list(filter(lambda x: x.find(group) != -1, experiments))
    for e in range(1):
        filtered_e = list(filter(lambda x: x.find(f'E{e}') != -1, filtered_group))
        for w in range(1):
            filtered_w = list(filter(lambda x: x.find(f'W{w}') != -1, filtered_e))
            for c in range(100):
                filtered_c = list(filter(lambda x: x.find(f'C{c}') != -1, filtered_w))

                agg = []
                for r in range(10):
                    filtered_r = list(filter(lambda x: x.find(f'R{r}') != -1, filtered_c))

                    if len(filtered_r) == 1:
                        experiment, = filtered_r

                        df = pd.DataFrame.from_records(pd.read_json(experiment)[prop])[func]
                        if reverse: df = df[len(df) - tasks]
                        else: df = df[0 + tasks]
                        name = os.path.basename(experiment)
                        data.update({name: df})
                        agg.append(df)

                if len(agg) > 0:
                    name = os.path.basename(os.path.dirname(group)).split('__')[0]

                    # total best
                    last = np.NINF
                    for i, entry in enumerate(agg):
                        sum = entry.sum(axis='rows')
                        if sum > last:
                            last = sum
                            df = entry
                            tag = i

                    # averaged best
                    # df = pd.concat(agg, axis='columns').aggregate('mean', axis='columns')
                    # tag = 'a'

                    name += f'E{e}-W{w}-C{c}-R{tag}'

                    sns.lineplot(df, alpha=0.75, label=name, ax=axs[0])
                    # sns.scatterplot(df, alpha=0.75, label=name, ax=axs[0])
                    sns.boxplot(x=len(data), y=df, label=name, ax=axs[1])
                    # sns.violinplot(x=len(data), y=df, label=name, ax=axs[1])

    axs[0].legend('')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)

    plt.suptitle(os.path.basename(os.path.dirname(group)))
    plt.tight_layout()
    plt.show()
'''

# '''
# accumulieren über runs, dann auch weniger daten
for group in sorted(groups):
    data = {}
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(19.2, 10.8))

    filtered_group = list(filter(lambda x: x.find(group) != -1, experiments))
    for e in range(1):
        filtered_e = list(filter(lambda x: x.find(f'E{e}') != -1, filtered_group))
        for w in range(1):
            filtered_w = list(filter(lambda x: x.find(f'W{w}') != -1, filtered_e))
            for c in range(10):
                filtered_c = list(filter(lambda x: x.find(f'C{c}') != -1, filtered_w))

                agg = []
                for r in range(10):
                    filtered_r = list(filter(lambda x: x.find(f'R{r}') != -1, filtered_c))

                    if len(filtered_r) == 1:
                        experiment, = filtered_r

                        df = pd.DataFrame.from_records(pd.read_json(experiment)[prop])[func]
                        if reverse: df = df[len(df) - tasks]
                        else: df = df[0 + tasks]
                        name = os.path.basename(experiment)
                        data.update({name: df})
                        agg.append(df)

                if len(agg) > 0:
                    name = os.path.basename(os.path.dirname(group)).split('__')[0]

                    # total best
                    last = np.NINF
                    for i, entry in enumerate(agg):
                        sum = entry.sum(axis='rows')
                        if sum > last:
                            last = sum
                            df = entry
                            tag = i

                    # averaged best
                    # df = pd.concat(agg, axis='columns').aggregate('mean', axis='columns')
                    # tag = 'a'

                    name += f'__E{e}-W{w}-C{c}-R{tag}'

                    print(name, '\t',
                        'T1:', np.round(np.array(df)[0], 3), '\t'
                        'T2:', np.round(np.array(df)[1], 3), '\t'
                        'T3:', np.round(np.array(df)[2], 3), '\t'
                        'TA:', np.round(np.average(df), 3), '\t'
                    )

                    sns.lineplot(df, alpha=0.75, label=name, ax=axs[0])
                    # sns.scatterplot(df, alpha=0.75, label=name, ax=axs[0])
                    sns.boxplot(x=len(data), y=df, label=name, ax=axs[1])
                    # sns.violinplot(x=len(data), y=df, label=name, ax=axs[1])

    axs[0].legend('')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)

    plt.suptitle(os.path.basename(os.path.dirname(group)))
    plt.tight_layout()
    plt.show()
# '''
