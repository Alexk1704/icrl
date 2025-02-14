'''
is it possible to use the __init__ again instead the initialize_once
or is another decorator function needed to flag the constructor (to be called only once per __new__)
'''


def singleton(cls):
    instances = {}

    def decorated(*args, **kwargs):
        if cls not in instances:
            instances.update({cls: cls(*args, **kwargs)})
        return instances[cls]

    return decorated
