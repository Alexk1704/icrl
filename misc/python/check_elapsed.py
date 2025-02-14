import re
import os
import sys
import glob
import json
import pandas
import datetime

import numpy as np
import matplotlib.pyplot as plt


if 'help' in sys.argv:
    print('Requires: <exp_path>.')
    sys.exit()


exp_path = sys.argv[1] if len(sys.argv) == 2 else './'

csv_files = glob.glob(os.path.join(exp_path, '**/db.csv'), recursive=True)
out_files = glob.glob(os.path.join(exp_path, '**/outputs/*.log'), recursive=True)


print('|^^^^^^^^^^^^^^^^^^^^^^^^^^^^|')
print('| Found a total of {:^4} DBs! |'.format(len(csv_files)))
print('|____________________________|')

for i, csv_file in enumerate(csv_files):
    choice = input(f'Process {csv_file} (y/n)? ')
    if choice in ['n', 'N']: continue

    print(f'Reading {i + 1}. DB...')
    DB = pandas.read_csv(csv_file)

    exp = DB['exp_id'][0].split('__')[0]
    print(f'Found {len(DB["exp_id"])} entries in DB!')

    exp_files = list(filter(lambda x: x.find(exp) != -1, out_files))
    print(f'Found {len(exp_files)} entries in out!')

    fig, axs = plt.subplots(2, 1)
    time_per_subtask = []
    steps_per_subtask = []

    axs[0].set_title('hours per subtask')
    axs[1].set_title('seconds per step')

    for exp_id, json_args in zip(DB['exp_id'], DB['json_args']):
        exp_id_files = list(filter(lambda x: x.find(exp_id) != -1, exp_files))

        with open(*exp_id_files, 'r') as fp:
            output = fp.read()

        ctx_switches = re.findall(r'(.*?Switch.*?track.*?)\n', output)
        try:
            timestamp_before = re.findall(r'(.*?timestamp before.*?)\n', output)[0]
            timestamp_after = re.findall(r'(.*?timestamp after.*?)\n', output)[0]

            timestamps = [datetime.datetime.strptime(entry[:23], '%Y-%m-%d %H:%M:%S,%f') for entry in ctx_switches]
            timestamps.append(datetime.datetime.fromisoformat(timestamp_after[-25:]).replace(tzinfo=None))
            time_per_subtask.append([x.total_seconds() for x in np.diff(timestamps)])

            train_steps = json.loads(json_args)['RLLibAgent']['training_duration']
            eval_steps = json.loads(json_args)['RLLibAgent']['evaluation_duration']
            steps_per_subtask.append([train_steps if entry.find('True') else eval_steps for entry in ctx_switches])
        except:
            print(f'Cannot calculate times from output file of {exp_id}!')

    try:
        time_data = np.full((len(time_per_subtask), max([len(entry) for entry in time_per_subtask])), np.nan)
        steps_data = np.full((len(steps_per_subtask), max([len(entry) for entry in steps_per_subtask])), np.nan)

        for i, (time_arr, steps_arr) in enumerate(zip(time_per_subtask, steps_per_subtask)):
            try:
                time_data[i] = time_arr
                steps_data[i] = steps_arr
            except: pass

        times = [d[m] for d, m in zip(time_data.T, ~np.isnan(time_data).T)]
        steps = [d[m] for d, m in zip(steps_data.T, ~np.isnan(steps_data).T)]

        axs[0].boxplot([np.divide(time, 60 * 60) for time in times], notch=True, showmeans=True)
        axs[1].violinplot([np.divide(time, step) for time, step in zip(times, steps)], showmeans=True, showmedians=True, showextrema=True)

        plt.show()
    except:
        print(f'Cannot create plot for experiment {exp}!')
