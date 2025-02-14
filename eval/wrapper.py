import os
import sys
import traceback


# custom invocation for testing
invocation = sys.argv

default_args = [
    '--root', '/home/dell/Desktop/Experiments/xxx/',
    '--exec', 'st',
    '--debug', 'no',
    '--draft', 'no',
    '--egge', 'no',
]

modules = [
    'samples',
    'buffer',
    'model',
    'policy',
    'agent',
]

basics = [
    '--format', 'png',
    '--mode', 'show',
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


for advanced in args_wrapper:
    invocation = [os.path.join(os.path.dirname(__file__), 'main.py'), *default_args]

    for module in modules:
        sys.argv = [*invocation, module, *basics, *advanced]
        print(f'Invoked: {sys.argv}')

        import main
        sys.exit()

        # try:
        #     import main
        #     # del main
        #     del sys.modules['main']
        # except Exception as e:
        #     traceback.print_exception(e)
