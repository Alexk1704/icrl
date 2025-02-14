import threading


def concurrent(cls):
    def decorated(*args, **kwargs):
        def call_async(cls):
            while cls.queue:
                cls.queue.pop(0)()

        cls.queue = []
        cls.call_async = call_async
        cls.thread = threading.Thread(target=cls.call_async)
        return cls

    return decorated

def async_call(func):
    def decorated(cls, *args, **kwargs):
        callee = func(cls, *args, **kwargs)
        cls.queue.append(lambda: callee)
        if not cls.thread.is_alive():
            cls.thread.start()
    return decorated


import time
import random


@concurrent
class Test():
    @staticmethod
    # @async_call
    def print_0(sleep):
        print('Hello staticmethod')
        time.sleep(sleep)

    @classmethod
    # @async_call
    def print_1(cls, sleep):
        print('Hello classmethod')
        time.sleep(sleep)

    # @async_call
    def print_2(self, sleep):
        print('Hello method')
        time.sleep(sleep)

print('aaa')
T = Test()
print('bbb')
for _ in range(10):
    rand = random.randint(0, 2)
    if rand == 0: T.print_0(random.random())
    elif rand == 1: T.print_1(random.random())
    elif rand == 2: T.print_2(random.random())

print('ccc')
