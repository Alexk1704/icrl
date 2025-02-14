import numpy as np

eval_repetition = 4
train_repetition = 1

eval_subtasks = 3
train_subtasks = 3

eval_steps = 10000
train_steps = 100000

total_steps = eval_steps * eval_subtasks * eval_repetition + train_steps * train_subtasks * train_repetition

sample = {
    'static': np.ndarray((1,), dtype=np.bool_),
    'random': np.ndarray((1,), dtype=np.bool_),
    'clock': np.ndarray((2,), dtype=np.float32),
    'state': np.ndarray((1,), dtype=np.float32),
    'action': np.ndarray((2,), dtype=np.float32),
    'reward': np.ndarray((1,), dtype=np.float32),
    'duration': np.ndarray((1,), dtype=np.float32),
}

pickle_overhead = 1024
list_overhead = 2048
numpy_overhead = 512

sample_size = 0
for key, value in sample.items():
    if value.dtype == np.bool_: sample_size += 8
    if value.dtype == np.float32: sample_size += 32
    sample_size += numpy_overhead

total_size = pickle_overhead + total_steps * (list_overhead + sample_size)

print('total_size (kb):', total_size / 1024)
print('total_size (mb):', total_size / 1024 / 1024)
print('total_size (gb):', total_size / 1024 / 1024 / 1024)


import os
import sys
import pickle

class Sample():
    def __init__(self, static=None, random=None, clock=None, state=None, action=None, reward=None, duration=None):
        self.static = static
        self.random = random
        self.clock = clock
        self.state = state
        self.action = action
        self.reward = reward
        self.duration = duration


class Step():
    def __init__(self):
        self.entries:list[Sample] = []

    def set(self, sample:Sample):
        self.entries.append(sample)

    def get(self, index) -> Sample:
        return self.entries[index]

    def len(self, accumulated=False):
        if accumulated: return len(self.entries)
        else: return len(self.entries)


class Episode():
    def __init__(self):
        self.entries:list[Step] = []

    def set(self, step:Step):
        self.entries.append(step)

    def get(self, index) -> Step:
        return self.entries[index]

    def len(self, accumulated=False):
        if accumulated: return sum([entry.len(True) for entry in self.entries])
        else: return len(self.entries)


class Trace():
    def __init__(self):
        self.entries:list[Episode] = []

    def set(self, episode:Episode):
        self.entries.append(episode)

    def get(self, index) -> Episode:
        return self.entries[index]

    def len(self, accumulated=False):
        if accumulated: return sum([entry.len(True) for entry in self.entries])
        else: return len(self.entries)


class History():
    def __init__(self):
        self.entries:list[Trace] = []

    def set(self, trace:Trace):
        self.entries.append(trace)

    def get(self, index) -> Trace:
        return self.entries[index]

    def len(self, accumulated=False):
        if accumulated: return sum([entry.len(True) for entry in self.entries])
        else: return len(self.entries)


class Wrapper():
    def __init__(self):
        self.entries:list[History] = []

    def set(self, history:History):
        self.entries.append(history)

    def get(self, index) -> History:
        return self.entries[index]

    def len(self, accumulated=False):
        if accumulated: return sum([entry.len(True) for entry in self.entries])
        else: return len(self.entries)

raw = Wrapper()
for _ in range(total_steps):
    H = History()
    T = Trace()
    E = Episode()
    S = Step()
    X = Sample(
        np.ndarray((1,), dtype=np.bool_),
        np.ndarray((1,), dtype=np.bool_),
        np.ndarray((2,), dtype=np.float32),
        np.ndarray((1,), dtype=np.float32),
        np.ndarray((2,), dtype=np.float32),
        np.ndarray((1,), dtype=np.float32),
        np.ndarray((1,), dtype=np.float32)
    )
    S.set(X)
    E.set(S)
    T.set(E)
    H.set(T)
    raw.set(H)

print(sys.getsizeof(raw))

with open('test.pkl', 'wb') as fp:
    pickle.dump(raw, fp)

file_stats = os.stat('test.pkl')
print('pickle_size (kb)', file_stats.st_size / 1024)
print('pickle_size (mb)', file_stats.st_size / 1024 / 1024)
print('pickle_size (gb)', file_stats.st_size / 1024 / 1024 / 1024)

os.remove('test.pkl')
