import re
import os
import sys
import glob
import json
import tqdm
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

    for exp_id, json_args in zip(DB['exp_id'], DB['json_args']):
        exp_id_files = list(filter(lambda x: x.find(exp_id) != -1, exp_files))

        with open(*exp_id_files, 'r') as fp:
            output = fp.read()

        ctx_switches = re.findall(r'(.*?Switch.*?track.*?)\n', output)

        train_subtasks = len(json.loads(json_args)['RLLibAgent']['train_subtasks'].split(' '))
        eval_subtasks = len(json.loads(json_args)['RLLibAgent']['eval_subtasks'].split(' '))
        context_change = json.loads(json_args)['RLLibAgent']['context_change']
        train_swap = int(json.loads(json_args)['RLLibAgent']['train_swap'])
        eval_swap = int(json.loads(json_args)['RLLibAgent']['eval_swap'])
        task_repetition = int(json.loads(json_args)['RLLibAgent']['task_repetition'])
        begin_with = json.loads(json_args)['RLLibAgent']['begin_with']
        end_with = json.loads(json_args)['RLLibAgent']['end_with']

        cnt = 0
        number_subtasks = 0

        if context_change == 'alternately':
            if train_swap == -1: train_swap = train_subtasks
            if eval_swap == -1: eval_swap = eval_subtasks
            while cnt // train_subtasks < task_repetition:
                number_subtasks += eval_swap
                number_subtasks += train_swap
                cnt += train_swap
            if begin_with == 'train': number_subtasks -= eval_subtasks
            if end_with == 'eval': number_subtasks += eval_subtasks

        if context_change == 'completely':
            train_swap = train_subtasks
            eval_swap = eval_subtasks
            while cnt // train_subtasks < task_repetition:
                number_subtasks += eval_swap
                number_subtasks += train_swap
                cnt += train_swap
            if begin_with == 'train': number_subtasks -= eval_subtasks
            if end_with == 'eval': number_subtasks += eval_subtasks

        progress = 0
        if len(ctx_switches) > 0:
            progress = len(ctx_switches) / number_subtasks

        width = 50
        completed = int(round(width * progress, 0))
        remaining = width - completed
        print('[{}]'.format('#' * completed + ' ' * remaining), exp_id, f'[{len(ctx_switches)}/{number_subtasks} subtask(s)]')
