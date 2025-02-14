import os
import sys
import glob
import traceback


# custom invocation for testing
invocation = sys.argv

EXPS = [
    *glob.glob('/mnt/STORAGE/TEMP/ICRL/**/results/success/*/'),
]

'''
EXPS = [
    # *glob.glob('/mnt/STORAGE/TEMP/ICRL/dqn-er-3tracks__24-06-25-00-27-21/results/success/*/'),
    # *glob.glob('/mnt/STORAGE/TEMP/ICRL/qgmm-3tracks__24-06-25-17-50-14/results/success/*/'),
    # *glob.glob('/mnt/STORAGE/TEMP/ICRL/dqn-3task__24-06-26-15-35-56/results/success/*/'),
    # *glob.glob('/mnt/STORAGE/TEMP/ICRL/qgmm-3task__24-06-26-15-35-33/results/success/*/'),
    # *glob.glob('/mnt/STORAGE/TEMP/ICRL/dqn-3task__24-06-27-23-24-58/results/success/*/'),
    # *glob.glob('/mnt/STORAGE/TEMP/ICRL/qgmm-3task__24-06-27-23-25-45/results/success/*/'),
    # *glob.glob('/mnt/STORAGE/TEMP/ICRL/dqn-base__24-06-27-23-28-12/results/success/*/'),
    # *glob.glob('/mnt/STORAGE/TEMP/ICRL/qgmm-base__24-06-27-23-27-31/results/success/*/'),
]
'''

filter_args = [
    # '--root', '/mnt/STORAGE/TEMP/ALEX/ALEX_EXPERIMENTE_alt/',
    # '--root', '/mnt/STORAGE/TEMP/ALEX/ALEX_EXPERIMENTE_neu/',
    # '--root', '/mnt/STORAGE/TEMP/ICRL/dqn-er-3tracks__24-06-25-00-27-21/',
    # '--root', '/mnt/STORAGE/TEMP/ICRL/qgmm-3tracks__24-06-25-17-50-14/',
    # '--root', '/mnt/STORAGE/TEMP/ICRL/dqn-3task__24-06-26-15-35-56/',
    # '--root', '/mnt/STORAGE/TEMP/ICRL/qgmm-3task__24-06-26-15-35-33/',
    # '--root', '/mnt/STORAGE/TEMP/ICRL/dqn-3task__24-06-27-23-24-58/',
    # '--root', '/mnt/STORAGE/TEMP/ICRL/qgmm-3task__24-06-27-23-25-45/',
    # '--root', '/mnt/STORAGE/TEMP/ICRL/dqn-base__24-06-27-23-28-12/',
    # '--root', '/mnt/STORAGE/TEMP/ICRL/qgmm-base__24-06-27-23-27-31/',
    # '--eval_selector', '0',
    # '--wrap_selector', '0',
    # '--comb_selector', '0',
    # '--run_selector', '0',
]

default_args = [
    '--exec', 'st',
    '--debug', 'no',
    '--draft', 'no',
    '--egge', 'no',
]

modules = [
    'samples',
    # 'buffer',
    # 'model',
    # 'policy',
    # 'agent',
]

basics = [
    '--format', 'png',
    '--mode', 'save',
    '--align', 'v',
    '--entity', '',
    '--plot', 'single',
    '--stack', 'runs',
    '--detail', 'step',
    '--aggregate', '',
]

args_wrapper = [
    # [
    #     '--category', 'categorical',
    #     '--type', 'pie',
    # ],
    # [
    #     '--category', 'distribution',
    #     '--type', 'violin',
    # ],
    [
        '--category', 'timeseries',
        '--type', 'curve',
    ],
    # [
    #     '--category', 'heatmap',
    #     '--type', 'curve',
    # ],
]

for exp in sorted(EXPS):
    for advanced in args_wrapper:
        invocation = [os.path.join(os.path.dirname(__file__), 'main.py'), '--root', exp, '--path', os.path.realpath(os.path.join(exp, '../../../evaluation')), *filter_args, *default_args]

        for module in modules:
            sys.argv = [*invocation, module, *basics, *advanced, '--name', os.path.basename(os.path.dirname(exp))]
            print(f'Invoked: {sys.argv}')

            # # HACK: hard exit after first module
            # import main
            # sys.exit()

            try:
                import main
                # del main
                del sys.modules['main']
            except Exception as e:
                traceback.print_exception(e)

'''
for advanced in args_wrapper:
    invocation = [os.path.join(os.path.dirname(__file__), 'main.py'), *filter_args, *default_args]

    for module in modules:
        sys.argv = [*invocation, module, *basics, *advanced]
        print(f'Invoked: {sys.argv}')

        # HACK: hard exit after first module
        import main
        sys.exit()

        # try:
        #     import main
        #     # del main
        #     del sys.modules['main']
        # except Exception as e:
        #     traceback.print_exception(e)
'''
