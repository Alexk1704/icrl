import sys
import time
import pprint
import ctypes
import weakref


from rllib_gazebo.utils.ArgsConfig import EntryArgs


class Cache():
    def __new__(cls):
        try: return cls.__instance__
        except:
            cls.initialize_once()
            cls.__instance__ = super().__new__(cls)
            return cls.__instance__

    @classmethod
    def initialize_once(cls):
        cls.available_triggers = [
            'step',
            'reset',
            'switch',
            'overall',
            'constructor', # (once - before)
            'destructor', # (once - after)
            'custom',
        ]

        cls.default_category = ''
        cls.default_trigger = 'switch'
        cls.default_counter = 0

        # internal vs. external
        cls.object_registry = {}

    @classmethod
    def register_object(cls, id, obj, category=None, trigger=None, counter=None):
        if category is None: category = cls.default_category
        if trigger is None: trigger = cls.default_trigger
        if counter is None: counter = cls.default_counter
        cls.object_registry[id] = (obj, category, trigger, counter)

    @classmethod
    def update_object(cls, id, obj=None, category=None, trigger=None, counter=None):
        entry = cls.object_registry[id]
        if obj is None: obj = entry[0]
        if category is None: category = entry[1]
        if trigger is None: trigger = entry[2]
        if counter is None: counter = entry[3]
        cls.object_registry.update({id: (obj, category, trigger, counter)})

    @classmethod
    def remove_object(cls, id):
        cls.object_registry.pop(id)

    @classmethod
    def by_category(cls, category=None):
        if category is None:
            return list(cls.object_registry.keys())
        else:
            bulk = []
            for key, value in cls.object_registry.items():
                if category == value[1]:
                    bulk.append(key)
            return bulk

    @classmethod
    def by_trigger(cls, trigger=None):
        if trigger is None:
            return list(cls.object_registry.keys())
        else:
            bulk = []
            for key, value in cls.object_registry.items():
                if trigger == value[2]:
                    bulk.append(key)
            return bulk

    @classmethod
    def by_counter(cls, counter=None):
        if counter is None:
            return list(cls.object_registry.keys())
        else:
            bulk = []
            for key, value in cls.object_registry.items():
                if counter == value[3]:
                    bulk.append(key)
            return bulk
