import re
import os
import sys
import glob
import pandas
import datetime

import numpy as np


if 'help' in sys.argv:
    print('Requires: <exp_path>.')
    sys.exit()


OK='\033[1;32m✓\033[0m'
ERR='\033[1;31m✗\033[0m'
WARN='\033[1;33m!\033[0m'
INFO='\033[1;34mi\033[0m'
UNK='\033[1;35m?\033[0m'
PART='\033[1;36m~\033[0m'


exp_path = sys.argv[1] if len(sys.argv) == 2 else './'

csv_files = glob.glob(os.path.join(exp_path, '**/db.csv'), recursive=True)
out_files = glob.glob(os.path.join(exp_path, '**/outputs/*.log'), recursive=True)
succ_files = glob.glob(os.path.join(exp_path, '**/results/success/*.tar.gz'), recursive=True)
fail_files = glob.glob(os.path.join(exp_path, '**/results/failed/*.tar.gz'), recursive=True)


def dynamic_bins(data, threshold=0.05):
    data = np.array(data)

    with_bin = []
    without_bin = np.argsort(data).tolist()

    while len(without_bin) > 0:
        with_bin.append([without_bin.pop(0)])

        offset = 0
        lower_bound = data[with_bin[-1]]
        upper_bound = lower_bound + np.multiply(data[with_bin[-1]], threshold)
        for i, not_assigned in enumerate(without_bin.copy()):
            if lower_bound <= data[not_assigned] <= upper_bound:
                with_bin[-1].append(without_bin.pop(i - offset))
                offset += 1

        '''
        overflow = []
        for i, not_assigned in enumerate(without_bin.copy()):
            bin_average = np.average(data[with_bin[-1]])
            if np.abs(np.diff([data[not_assigned], bin_average])) < np.multiply(bin_average, threshold):
                with_bin[-1].append(without_bin.pop(i))

                for j, assigned in enumerate(with_bin[-1].copy()):
                    bin_average = np.average(data[with_bin[-1]])
                    if np.abs(np.diff([data[assigned], bin_average])) < np.multiply(bin_average, threshold):
                        overflow.append(with_bin[-1].pop(j))

        with_bin.insert(-1, overflow)
        '''

    '''
    for bin in with_bin:
        stats = {
            'len': len(bin),
            'min': np.min(data[bin]),
            'max': np.max(data[bin]),
            'ptp': np.ptp(data[bin]),
            'median': np.median(data[bin]),
            'mean': np.mean(data[bin]),
            'dev': np.divide(np.ptp(data[bin]), np.mean(data[bin])),
        }
        print(stats)

        for uniq, cnt in zip(*np.unique(data[bin], return_counts=True)):
            print(uniq, f'{cnt}x')
    '''

    return with_bin


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

    filtered_out_files = list(filter(lambda x: x.find(exp) != -1, out_files))
    filtered_succ_files = list(filter(lambda x: x.find(exp) != -1, succ_files))
    filtered_fail_files = list(filter(lambda x: x.find(exp) != -1, fail_files))

    print(f'Found {len(filtered_out_files)} entries in outputs!')
    print(f'Found {len(filtered_succ_files)} entries in results/success!')
    print(f'Found {len(filtered_fail_files)} entries in results/failed!')

    print('| {:1} | {:^25} | {:^5} | {:^5} | {:^5} |'.format('', 'name', 'out', 'succ', 'fail'))
    print('| {:1} | {:^25} | {:^5} | {:^5} | {:^5} |'.format('-' * 1, '-' * 25, '-' * 5, '-' * 5, '-' * 5))
    for exp_id in DB['exp_id']:
        stats = {'out': False, 'succ': False, 'fail': False}

        if '\n'.join(filtered_out_files).find(exp_id) != -1: stats['out'] = True
        if '\n'.join(filtered_succ_files).find(exp_id) != -1: stats['succ'] = True
        if '\n'.join(filtered_fail_files).find(exp_id) != -1: stats['fail'] = True

        if stats['out']:
            if stats['succ']: symbol = OK
            else:
                if stats['fail']: symbol = INFO
                else: symbol = WARN
        else:
            if stats['succ']: symbol = PART
            else:
                if stats['fail']: symbol = UNK
                else: symbol = ERR
        stats_frmt = ' | '.join(['{:^5}'.format('###' if value else '') for value in stats.values()])
        print('| {:1} | {:25} | {} |'.format(symbol, exp_id, stats_frmt))

    out_file_stats = {out_file: os.stat(out_file) for out_file in filtered_out_files}
    succ_file_stats = {succ_file: os.stat(succ_file) for succ_file in filtered_succ_files}
    fail_file_stats = {fail_file: os.stat(fail_file) for fail_file in filtered_fail_files}

    out_bins = dynamic_bins([value.st_size for value in out_file_stats.values()])
    succ_bins = dynamic_bins([value.st_size for value in succ_file_stats.values()])
    fail_bins = dynamic_bins([value.st_size for value in fail_file_stats.values()])

    print(f'Estimated {len(out_bins)} possible log groups.')
    print(f'Estimated {len(succ_bins)} possible succ groups.')
    print(f'Estimated {len(fail_bins)} possible fail groups.')

    choice = input('Inspect details (y/n)? ')
    if choice in ['n', 'N']: continue

    for i, out_bin in enumerate(out_bins):
        details = {
            'lines': [],
            'hosts': [],
            'start': [],
            'finish': [],
        }
        for index in out_bin:
            with open(filtered_out_files[index], 'r') as fp:
                ctx = fp.read()
                # ctx = fp.readlines() # has \n etc.
            ctx = ctx.splitlines()

            details['lines'].append(len(ctx))

            for j, line in enumerate(ctx):
                if line.find('host is') != -1: details['hosts'].append(line.split(' ')[-1])
                if line.find('timestamp before') != -1: details['start'].append(ctx[j + 1])
                if line.find('timestamp after') != -1: details['finish'].append(ctx[j + 1])

        avg_lines = int(np.round(np.average(details['lines']), 0))

        hosts, modes = np.unique(details['hosts'], return_counts=True)
        freq_hosts = [f'{host} {mode}x' for host, mode in zip(hosts, modes)]

        durations = []
        for start, finish in zip(details['start'], details['finish']):
            try:
                duration = datetime.datetime.fromisoformat(finish) - datetime.datetime.fromisoformat(start)
                durations.append(duration.total_seconds() / 60 / 60)
            except: pass
        if durations:
            avg_duration = np.round(np.average(durations), 2)

        print(f'{i + 1}. log group has {len(out_bin)} members:')
        print(f'\tavg length: ~ {avg_lines} lines')
        print(f'\tfreq hosts: {freq_hosts}')
        if len(durations) > 0: print(f'\tavg duration: {avg_duration} hours')

        choice = input('Which log should be returned (all, min, med, max, rand)? ')
        match choice:
            case 'all': indices = out_bin
            case 'min': indices = [out_bin[0]]
            case 'med': indices = [out_bin[len(out_bin) // 2]]
            case 'max': indices = [out_bin[-1]]
            case _: indices = [np.random.choice(out_bin)]

        for index in indices:
            print(f'\t{filtered_out_files[index]}')
