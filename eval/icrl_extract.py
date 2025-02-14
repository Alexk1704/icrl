import os
import sys
import glob
import shutil


# EXP = 'E0-W0-C1-R2'
# EXP = 'E0-W0-C3-R0'
# EXP = 'E0-W0-C5-R1'

# GRP = 'dqn-base'

# ---

# EXP = 'E0-W0-C1-R1'
# EXP = 'E0-W0-C3-R2'
EXP = 'E0-W0-C5-R0'

GRP = 'qgmm-base'


experiments = glob.glob(f'/mnt/STORAGE/TEMP/ICRL/{GRP}*/evaluation/samples/**/*', recursive=True)
path = os.path.join('./', '__'.join([GRP, EXP]))

if not os.path.exists(path):
    os.makedirs(path)

for experiment in experiments:
    if GRP in experiment and EXP in experiment:
        if '.json' in experiment: shutil.copyfile(experiment, os.path.join(path, 'stats.json'))
        for plot in ['s', 'sd', 'a', 'ad', 'r', 'rd', 'rs', 'rl', 're']:
            if f'/{plot}/' in experiment: shutil.copyfile(experiment, os.path.join(path, f'{plot}.png'))
