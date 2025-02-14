import pprint
import numpy as np


nodes = 20

amount = {
    'grps': 2,
    'exps': 10,
    'runs': 3,
}

stats = {
    'train': {
        'steps': {
            'tasks': 3,
            'iterations': 100000,
            'repetition': 1,
        },
        'times': {
            'step': 0.7,
            'reset': 0.3,
            'scale': 5e-6,
        }
    },
    'eval': {
        'steps': {
            'tasks': 3,
            'iterations': 10000,
            'repetition': 3,
        },
        'times': {
            'step': 0.7,
            'reset': 0.3,
            'scale': 5e-6,
        }
    }
}

pprint.pprint(amount)
pprint.pprint(stats)
print('\n---\n')

executions = np.prod(list(amount.values()))

steps = {}
times = {}
for key, value in stats.items():
    lower_bound = value['times']['step']
    upper_bound = np.add(lower_bound, value['times']['reset'])

    steps[key] = np.prod(list(value['steps'].values()))
    times[key] = np.multiply([lower_bound, upper_bound], steps[key]) + np.sum(np.multiply(value['times']['scale'], np.arange(steps[key])))

for key, value in stats.items():
    print('{:5} steps per run:\t{:12.0f} ({:5.2%})'.format(key, steps[key], np.divide(steps[key], np.sum(list(steps.values()), axis=0))))
print()

steps_per_run = np.sum(list(steps.values()), axis=0)
steps_per_exp = np.multiply(steps_per_run, amount['runs'])
steps_per_grp = np.multiply(steps_per_exp, amount['exps'])
total_steps   = np.multiply(steps_per_grp, amount['grps'])

print('steps_per_run:\t\t{:12.0f}'.format(steps_per_run))
print('steps_per_exp:\t\t{:12.0f}'.format(steps_per_exp))
print('steps_per_grp:\t\t{:12.0f}'.format(steps_per_grp))
print('total_steps:  \t\t{:12.0f}'.format(total_steps))
print('\n---\n')

# ---

time = np.divide(np.sum(list(times.values()), axis=0), steps_per_run)
time_unit = np.prod([60, 60])

for key, value in stats.items():
    single_time = np.divide(np.multiply(steps[key], time), time_unit)
    total_time = np.divide(np.multiply(steps_per_run, time), time_unit)
    print('{:5} time per run: \t[{:12.3f}h, {:12.3f}h] ({:5.2%} - {:5.2%})'.format(key, *single_time, *np.divide(single_time, total_time)))
print()

time_per_run = np.divide(np.multiply(steps_per_run, time), time_unit)
time_per_exp = np.divide(np.multiply(steps_per_exp, time), time_unit)
time_per_grp = np.divide(np.multiply(steps_per_grp, time), time_unit)
total_time   = np.divide(np.multiply(total_steps, time), time_unit)

print('time_per_run:\t\t[{:12.3f}h, {:12.3f}h]'.format(*time_per_run))
print('time_per_exp:\t\t[{:12.3f}h, {:12.3f}h]'.format(*time_per_exp))
print('time_per_grp:\t\t[{:12.3f}h, {:12.3f}h]'.format(*time_per_grp))
print('total_time:  \t\t[{:12.3f}h, {:12.3f}h]'.format(*total_time))
print()

time_factor = np.ceil(np.divide(executions, nodes))

print('by using {:3} nodes some of them have at least {:3.0f} executions'.format(nodes, time_factor))
print('=> {:12.3f}h - {:12.3f}h on the cluster'.format(*np.multiply(time_per_run, time_factor)))
print('\n---\n')

# ---

header = 8 * 512
overhead = 8 * 64

variables = {
    'static': (1, 1),
    'random': (1, 1),
    'clock': (2, 32),
    'state': (1, 32),
    'action': (2, 32),
    'reward': (1, 32),
    'duration': (1, 32),
}

size = np.sum([np.prod(variable) for variable in variables.values()]) + overhead
size_unit = np.prod([8, 1024, 1024])

for key, value in stats.items():
    single_size = np.divide(np.multiply(steps[key], size), size_unit)
    total_size = np.divide(np.multiply(steps_per_run, size), size_unit)
    print('{:5} size per run: \t{:12.3f}MB ({:5.2%})'.format(key, single_size, np.divide(single_size, total_size)))
print()

size_per_run = np.divide(np.add(header, np.multiply(steps_per_run, size)), size_unit)
size_per_exp = np.divide(np.add(header, np.multiply(steps_per_exp, size)), size_unit)
size_per_grp = np.divide(np.add(header, np.multiply(steps_per_grp, size)), size_unit)
total_size   = np.divide(np.add(header, np.multiply(total_steps, size)), size_unit)

print('sizes_per_run:\t\t{:12.3f}MB'.format(size_per_run))
print('sizes_per_exp:\t\t{:12.3f}MB'.format(size_per_exp))
print('sizes_per_grp:\t\t{:12.3f}MB'.format(size_per_grp))
print('total_sizes:  \t\t{:12.3f}MB'.format(total_size))
print()

compression_rate = 0.8

print('by packing at least with a compression rate about {:.3f}'.format(compression_rate))
print('=> {:12.3f}MB on the drive'.format(np.multiply(total_size, compression_rate)))
print()
