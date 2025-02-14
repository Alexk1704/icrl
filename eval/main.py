import re
import os
import sys
import time
import copy
import json
import morph
import pprint
import argparse

import pandas as pd

from plots import samples, buffer, model, policy, agent
from utils import tools

''' Usage:
export PYTHONPATH=$PYTHONPATH:$GIT_PATH/sccl/src
export PYTHONPATH=$PYTHONPATH:$GIT_PATH/icrl/workspace/devel/src/rllib_gazebo

Local usage for a single results directory:
python3 $GIT_PATH/icrl/eval/main.py --root /tmp/ak/rllib_gazebo/dat/default__E0-W0-C0-R0 --egge no

Usage for generated experiment structure from EGGE:
python3 $GIT_PATH/icrl/eval/main.py --root ~/Desktop/VALIDATE__24-03-19-14-18-58/ --state success --wrap_selector 0 --comb_selector 0 1
'''

''' Returns:
EVAL_ID: { WRAP_ID: { COMB_ID: { RUN_ID: {
    'info' : {
        'exp_id': str(EXP_ID),
        'json_args': dict(JSON_ARGS),
        'tasks': list(TASKS),
        'counters': list(COUNTERS),
        'paths': dict(PATHS)
    },
    'raw': [{
        'subtask': {
            'mode': eval,
            'name': straight,
            'total_steps': 42,
            'total_resets': 1337,
            'total_switches': 1,
        },
        'data': {
            'q_values': q_values.pkl,
            'samples': samples.pkl,
            'spawns': spawns.pkl,
            'stats': stats.pkl
        },
        'debug': {
            'odo': odo.pkl,
        },
        'protos': {
            'xyz': .npy
        }
    }]
}}}}
'''

modules = [samples, buffer, model, policy, agent]
plots = {module.__name__.split('.')[-1]: module for module in modules}

args_lookup = {}

parser = argparse.ArgumentParser()
parser.add_argument('--root',                      type=str, default='./',                                         help='directory to evaluate')
parser.add_argument('--exec',                      type=str, default='mt',      choices=['st', 'mt'],              help='only single experiment')
parser.add_argument('--debug',                     type=str, default='no',      choices=['yes', 'no'],             help='enable debugging prints')
parser.add_argument('--draft',                     type=str, default='yes',     choices=['yes', 'no'],             help='try to reduce duration')
parser.add_argument('--egge',                      type=str, default='yes',     choices=['yes', 'no'],             help='only single experiment')
parser.add_argument('--state',                     type=str, default='success', choices=['', 'success', 'failed'], help='return code of results')
parser.add_argument('--path',                      type=str, default='./eval_plots/',                              help='save location of plots')
parser.add_argument('--pattern',                   type=str, default='%n_%c',                                      help='naming pattern of files')
parser.add_argument('--eval_selector',  nargs='+', type=int, default=[],                                           help='select on a eval-level')
parser.add_argument('--wrap_selector',  nargs='+', type=int, default=[],                                           help='select on a wrap-level')
parser.add_argument('--comb_selector',  nargs='+', type=int, default=[],                                           help='select on a comb-level')
parser.add_argument('--run_selector',   nargs='+', type=int, default=[],                                           help='select on a run-level')

args_lookup.update({'default': parser})

subparser = argparse.ArgumentParser(add_help=False)
subparser.add_argument('--name',               type=str,   default='',                                                            help='name of the figure')
subparser.add_argument('--title',              type=str,   default='',                                                            help='title of the figure')
subparser.add_argument('--size',      nargs=2, type=float, default=[19.2, 12],                                                    help='size of the figure')
subparser.add_argument('--format',             type=str,   default='png',    choices=['png', 'svg', 'pdf'],                       help='format of the figure')
subparser.add_argument('--mode',               type=str,   default='show',   choices=['show', 'save'],                            help='show or save plots')
subparser.add_argument('--style',              type=str,   default='dark',   choices=['default', 'xkcd', 'white', 'dark'],        help='style of the figure')
subparser.add_argument('--align',              type=str,   default='h',      choices=['r', 'h', 'c', 'v'],                        help='alignment of plots')
subparser.add_argument('--entity',             type=str,   default='',                                                            help='entities to plot')
subparser.add_argument('--plot',               type=str,   default='single', choices=['single', 'multiple'],                      help='integrate or separate')
subparser.add_argument('--stack',              type=str,   default='runs',   choices=['exps', 'runs', 'combs', 'wraps', 'evals'], help='stack values over eval')
subparser.add_argument('--detail',             type=str,   default='step',   choices=['step', 'reset', 'switch', 'overall'],      help='combine values over time')
subparser.add_argument('--aggregate',          type=str,   default='',                                                            help='aggregate the plot')
subparser.add_argument('--category',           type=str,   default='',                                                            help='categories to plot') # frequency, timeseries, etc.
subparser.add_argument('--type',               type=str,   default='',                                                            help='the type of plot') # scatter, violin etc.
# metriken, kennzahlen und abhängigkeiten
# scatter, heatmap, relationen/zusammenhänge
#
# measures, skalenniveaus, uni-/multivariat
# lagemaße, streuungsmaße, zusammenhangsmaße
#
# aggregations and smoothing
# moving (windowed) average

nested = parser.add_subparsers()
for key, value in plots.items():
    subsubparser = nested.add_parser(key, parents=[subparser, value.parser], help=f'{key} -h, --help')
    args_lookup.update({key: subsubparser})

# HACK
parser.parse_known_args()

matches = [len(sys.argv)]
for plot in plots.keys():
    try: matches.append(sys.argv.index(plot))
    except: pass
matches.sort()

for key, value in args_lookup.items():
    if key == 'default':
        args_lookup[key] = value.parse_known_args(sys.argv[1:min(matches)])[0]
    else:
        try:
            index = sys.argv.index(key)
            window = sys.argv[index:matches[matches.index(index) + 1]]
            args_lookup[key] = value.parse_known_args(window)[0]
        except: args_lookup[key] = None


def resolve_eval_data(path):
    print(f'Looking for experiments in {path}.')

    entries = tools.resolve_root(path)
    print(f'Found {len(entries)} entries.')

    dirs = tools.return_dirs(entries)
    print(f'Found {len(dirs)} dirs.')

    files = tools.return_files(entries)
    print(f'Found {len(files)} files')

    files_by_type = tools.resolve_filetypes(files)
    print(f'Found {len(files_by_type)} file types ({list(files_by_type.keys())}).')

    file_types = ['.tar.gz']
    archives = tools.filter_filetypes(files_by_type, file_types)
    print(f'Found {len(archives)} archives of type {file_types}.')

    for archive in archives:
        try:
            print(f'Try to unpack {archive}')
            tools.unpack_archive(archive)
        except Exception as e:
            print(e)

    return entries, dirs, files, files_by_type

def process_db_entries(csv_files):
    if os.path.basename(file) == 'db.csv':
        db = pd.read_csv(file, converters={'json_args': json.loads})

        entries = {
            entry['eval_id']: {
                entry['wrap_id']: {
                    entry['comb_id']: {
                        entry['run_id']: {
                            'info': {
                                'exp_id': entry['exp_id'],
                                'json_args': entry['json_args'],
                            }
                        }
                    }
                }
            }
        for _, entry in db.iterrows()}

        return entries

def filter_evaluation(entries, eval_selector, wrap_selector, comb_selector, run_selector):
    for eval_key, eval_dict in copy.deepcopy(entries).items():
        if eval_key in eval_selector:
            for wrap_key, wrap_dict in eval_dict.items():
                if wrap_key in wrap_selector:
                    for comb_key, comb_dict in wrap_dict.items():
                        if comb_key not in comb_selector:
                            for run_key, _ in comb_dict.items():
                                if run_key in run_selector:
                                    pass
                                else:
                                    entries[eval_key][wrap_key][comb_key].pop(run_key)
                                    print('Removed experiment with run_id:', run_key)
                        else:
                            entries[eval_key][wrap_key].pop(comb_key)
                            print('Removed experiment with comb_id:', comb_key)
                else:
                    entries[eval_key].pop(wrap_key)
                    print('Removed experiment with wrap_id:', wrap_key)
        else:
            entries.pop(eval_key)
            print('Removed experiment with eval_id:', eval_key)

    return entries

def collect_experiments(entries, egge, dir_path, state):
    for eval_key, eval_dict in copy.deepcopy(entries).items():
        for wrap_key, wrap_dict in eval_dict.items():
            for comb_key, comb_dict in wrap_dict.items():
                for run_key, run_dict in comb_dict.items():
                    if egge == 'yes':
                        exp_id = run_dict['info']['exp_id']
                        base_path = os.path.join(dir_path, 'results', state, exp_id)

                    if egge == 'no':
                        base_path = os.path.dirname(dir_path)
                        exp_id = base_path.split('/')[-1]

                    if len(run_dict) == 0:
                        json_args = {}
                        try:
                            json_args.update({'FRONTEND': tools.load_file(*tools.resolve_pattern(os.path.join(dir_path, 'cfgs'), '**/FRONTEND.*'))})
                            json_args.update({'BACKEND': tools.load_file(*tools.resolve_pattern(os.path.join(dir_path, 'cfgs'), '**/BACKEND.*'))})
                        except: pass
                        temp = {'info': {'exp_id': exp_id, 'json_args': json_args}}
                        entries[eval_key][wrap_key][comb_key][run_key] = temp

                    lookup = {}
                    for file in sorted(tools.resolve_pattern(os.path.join(base_path, 'infos'), '**/*.*')):
                        try:
                            name = os.path.basename(file).split('.', 1)[0]
                            lookup.update({name: tools.load_file(file)})
                        except: print(f'Loading {name} failed for: {file}')

                    try:
                        info = {
                            'tasks': lookup['tasks'],
                            'counters': lookup['counters'],
                            'paths': {name: os.path.join(base_path, name) for name in ['data', 'debug', 'protos']},
                        }
                        raw = resolve_experiment_data(info, lookup)
                        entries[eval_key][wrap_key][comb_key][run_key]['info'].update(info)
                        entries[eval_key][wrap_key][comb_key][run_key]['raw'] = raw
                    except:
                        print(f'Skipping experiment {exp_id}...')
                        entries[eval_key][wrap_key][comb_key].pop(run_key)

    return entries

def resolve_experiment_data(info, lookup):
    def resolve_structs(location, level, counter):
        levels = ['step', 'reset', 'switch', 'overall']
        for file in sorted(tools.resolve_pattern(location, '**/*.*')):
            try:
                path = os.path.relpath(file, location)
                index = int(*re.findall(r'\d+', path))

                if index == counter:
                    path = path.lower().split('.', 1)[0]
                    name = '.'.join(filter(lambda x: x != '' and re.match('|'.join(levels + [str(counter)]), x) == None, path.split('/')))

                    yield name, tools.load_file(file)
            except: print(f'Data cannot be resolved: {file}')

    temp = []
    for i, (task, counter) in enumerate(zip(info['tasks'], info['counters'])):
        subtask = {
            'mode': task['mode'],
            'name': task['name'],
            'total_steps': sum(counter),
            'total_resets': len(counter),
            'total_switches': 1,
        }

        # maybe unflatten, if needed?!
        temp.append({
            'subtask': subtask,
            **{
                name: dict(resolve_structs(location, 'switch', i))
                for name, location in info['paths'].items()
            }
        })

    return temp

'''
def resolve_experiment_data(info, lookup):
    def resolve_structs(location, level):
        levels = ['step', 'reset', 'switch', 'overall']
        for file in sorted(tools.resolve_pattern(location, '**/*.*')):
            try:
                path = os.path.relpath(file, location)
                index = re.findall(r'\d+', path)

                path = path.lower().split('.', 1)[0]
                name = '.'.join(filter(lambda x: x != '' and re.match('|'.join(levels + index), x) == None, path.split('/')))

                yield ([int(i) for i in index], {name: tools.load_file(file)})
            except: print(f'Data cannot be resolved: {file}')

    subtasks = [{
        'mode': task['mode'],
        'name': task['name'],
        'total_steps': sum(counter),
        'total_resets': len(counter),
        'total_switches': 1,
    } for task, counter in zip(info['tasks'], info['counters'])]

    objects = {}
    for name, location in info['paths'].items():
        merged = {}
        for index, entry in resolve_structs(location, 'switch'):
            try: merged[tuple(index)].update(entry)
            except: merged[tuple(index)] = entry
        objects[name] = [merged[index] for index in sorted(merged)]

    return {'subtasks': pd.DataFrame(subtasks), **{key: pd.DataFrame(value) for key, value in objects.items()}}
'''

def debug(data):
    def extract_keys(keys, index):
        try: key_levels[f'level_{index}'].update(set(keys))
        except: key_levels[f'level_{index}'] = set(keys)

    def resolve_keys(data, level=0):
        if isinstance(data, dict):
            data.pop('json_args', None)
            extract_keys(data.keys(), level)
            for value in data.values():
                resolve_keys(value, level + 1)
        elif isinstance(data, list):
            for value in data:
                resolve_keys(value, level)

    key_levels = {}
    resolve_keys(copy.deepcopy(data))
    return key_levels


# if __name__ == '__main__':
wrapper = []
args = args_lookup['default']

if args.debug == 'yes':
    print('lookup dict:')
    pprint.pprint(args_lookup)

if args.egge == 'yes': # E-W-C-R with db-file
    entries, dirs, files, files_by_type = resolve_eval_data(args.root)

    for file in files_by_type['csv']:
        print('Next file:', file)

        choice = input('Continue (y/n)? ')
        if choice in ['y', 'Y']: pass
        elif choice in ['n', 'N']: continue
        else: sys.exit()

        print('processing data')
        start_time = time.time()
        exps = process_db_entries(file)
        exps = filter_evaluation(exps, args.eval_selector, args.wrap_selector, args.comb_selector, args.run_selector)
        exps = collect_experiments(exps, args.egge, args.root, args.state)
        end_time = time.time()
        print('elapsed time: {:.3f}'.format(end_time - start_time))

        wrapper.append(exps)

if args.egge == 'no': # E-W-C-R without db-file
    print('processing data')
    start_time = time.time()
    exps = {'0': {'0': {'0': {'0': {}}}}}
    exps = collect_experiments(exps, args.egge, args.root, args.state)
    end_time = time.time()
    print('elapsed time: {:.3f}'.format(end_time - start_time))

    wrapper = [exps]

for exps in wrapper:
    if args.debug == 'yes':
        print('keys dict:')
        pprint.pprint(debug(exps))

    for key, value in plots.items():
        if args_lookup[key] != None:
            print(f'invoke entry {key}')
            start_time = time.time()
            value.entry(exps, {**vars(args), **vars(args_lookup[key]), 'module': key})
            end_time = time.time()
            print('elapsed time: {:.3f}'.format(end_time - start_time))
