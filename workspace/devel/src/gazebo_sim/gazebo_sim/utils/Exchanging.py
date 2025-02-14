import os
import sys
import gzip
import json
import yaml
import numpy
import pickle
import shelve
import tarfile


from gazebo_sim.utils.Caching import Cache
from gazebo_sim.utils.ArgsConfig import Args


'''
use this class for internal state-full checkpointing != RLLib or TF
'''

class Exchanger():
    def __new__(cls):
        try: return cls.__instance__
        except:
            cls.initialize_once()
            cls.__instance__ = super().__new__(cls)
            return cls.__instance__

    @classmethod
    def initialize_once(cls):
        cls.supported = {
            'archive': {
                'tar': {'open': tarfile.open, 'binary': False},
                'gz':  {'open': gzip.open,    'binary': True},
            },
            'format': {
                'npy':  {'dump': numpy.save,  'load': numpy.load,  'binary': True},
                'pkl':  {'dump': pickle.dump, 'load': pickle.load, 'binary': True},
                'json': {'dump': json.dump,   'load': json.load,   'binary': False},
                'yaml': {'dump': yaml.dump,   'load': yaml.load,   'binary': False},
            }
        }

        # HACK
        cls.bulk_counters = {
            'step': 0,
            'reset': 0,
            'switch': 0,
            'overall': 0,
        }

        cls.base_path = os.path.join(Args().root_dir, Args().exp_id)

        cls.path_register = {}

        cls.default_names = ['config', 'info', 'log', 'stat', 'data', 'debug', 'proto', 'checkpoint']
        cls.default_paths = ['cfgs', 'infos', 'logs', 'stats', 'data', 'debug', 'protos', 'ckpts']

        # stateful objects
        for name, path in zip(cls.default_names, cls.default_paths):
            cls.add_path(name, path, cls.base_path)

        for path in cls.path_register.values():
            cls.ensure_path(path)

        # HACK: is here misplaced
        cls.store(vars(Args().args), f'{Args().ENTRY}.json', cls.path_register['config'])

    @classmethod
    def add_path(cls, name, path, base_path=None):
        if base_path is None: base_path = cls.base_path
        path = os.path.join(base_path, path)

        cls.path_register.update({name: path})

    @classmethod
    def remove_path(cls, name):
        cls.path_register.pop(name)

    @classmethod
    def resolve(cls, extensions, lookup):
        for ext in reversed(extensions):
            if ext in cls.supported[lookup]:
                return cls.supported[lookup][ext]

    @classmethod
    def ensure_path(cls, path_tree):
        if not os.path.exists(path_tree):
            os.makedirs(path_tree)

    @classmethod
    def store(cls, obj, file_name, base_path=None):
        if base_path is None: base_path = cls.base_path
        file_path = os.path.join(base_path, file_name)

        extensions = os.path.basename(file_path).split('.')[1:]
        archive = cls.resolve(extensions, 'archive')
        format = cls.resolve(extensions, 'format')

        cls.ensure_path(os.path.dirname(file_path))

        if archive is not None and format is not None:
            mode = 'wb' if archive['binary'] else 'w'
            with format['open'](file_path, mode) as fp:
                try: format['dump'](obj, fp)
                except: format['dump'](fp, obj)
        if archive is None and format is not None:
            mode = 'wb' if format['binary'] else 'w'
            with open(file_path, mode) as fp:
                try: format['dump'](obj, fp)
                except: format['dump'](fp, obj)

    @classmethod
    def restore(cls, file_name, base_path=None):
        if base_path is None: base_path = cls.base_path
        file_path = os.path.join(base_path, file_name)

        extensions = os.path.basename(file_path).split('.')[1:]
        archive = cls.resolve(extensions, 'archive')
        format = cls.resolve(extensions, 'format')

        if archive is not None and format is not None:
            mode = 'rb' if archive['binary'] else 'r'
            with format['open'](file_path, mode) as fp:
                return format['load'](fp)
        if archive is None and format is not None:
            mode = 'rb' if format['binary'] else 'r'
            with open(file_path, mode) as fp:
                return format['load'](fp)

    @classmethod
    def bulk_store(cls, category=None, trigger=None, counter=None):
        ids = []
        if category is not None: ids.append(Cache().by_category(category))
        if trigger is not None: ids.append(Cache().by_trigger(trigger))
        if counter is not None: ids.append(Cache().by_counter(counter))

        try:
            # HACK
            bulk_counter = cls.bulk_counters[trigger]
            for id in set.intersection(*map(lambda x: set(x), ids)):
                obj, _category, _trigger, _counter = Cache().object_registry[id]
                if obj is not None:
                    cls.store(obj, f'{_trigger}_{bulk_counter}.pkl', os.path.join(cls.path_register[_category], id))
                    Cache().object_registry[id] = (None, _category, _trigger, _counter) # HACK
                    Cache().update_object(id, counter=_counter + 1)
            cls.bulk_counters[trigger] += 1
        except:
            for id in set.intersection(*map(lambda x: set(x), ids)):
                obj, _category, _trigger, _counter = Cache().object_registry[id]
                if obj is not None:
                    cls.store(obj, f'{_trigger}_{_counter}.pkl', os.path.join(cls.path_register[_category], id))
                    Cache().object_registry[id] = (None, _category, _trigger, _counter) # HACK
                    Cache().update_object(id, counter=_counter + 1)

    @classmethod
    def bulk_restore(cls, category=None, trigger=None, counter=None):
        ids = []
        if category is not None: ids.append(Cache().by_category(category))
        if trigger is not None: ids.append(Cache().by_trigger(trigger))
        if counter is not None: ids.append(Cache().by_counter(counter))

        try:
            # HACK
            bulk_counter = cls.bulk_counters[trigger]
            for id in set.intersection(*map(lambda x: set(x), ids)):
                obj, _category, _trigger, _counter = Cache().object_registry[id]
                try: _obj = cls.restore(f'{_trigger}_{bulk_counter}.pkl', os.path.join(cls.path_register[_category], id))
                except: _obj = None
                Cache().update_object(id, obj=_obj)
        except:
            for id in set.intersection(*map(lambda x: set(x), ids)):
                obj, _category, _trigger, _counter = Cache().object_registry[id]
                try: _obj = cls.restore(f'{_trigger}_{_counter}.pkl', os.path.join(cls.path_register[_category], id))
                except: _obj = None
                Cache().update_object(id, obj=_obj)
