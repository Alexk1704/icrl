'''
wrapper = combines multiple histories (run)
    is a own object which contains histories
history = all traces over all tasks (experiment)
    is a own object which contains traces
trace = all episodes within a subtask (task)
    is a own object which contains episodes
episode = all mini-batches within a try (trajectory)
    is a own object which contains mini-batches
step = all samples within a single update (iteration)
    is a own object which contains samples (usually 1)

sample = is the smallest data instance of a control cycle
    is a own object which contains all states (object), actions (object), rewards (array) etc.
state
    is a numpy array which contains the data
action
    is a numpy array which contains the data
reward
    is a numpy array which holds a value
'''

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


class Evaluator():
    def __new__(cls):
        try: return cls.__instance__
        except:
            cls.initialize_once()
            cls.__instance__ = super().__new__(cls)
            return cls.__instance__

    @classmethod
    def initialize_once(cls):
        cls.WRAPPER = Wrapper
        cls.HISTORY = History
        cls.TRACE = Trace
        cls.EPISODE = Episode
        cls.STEP = Step
        cls.SAMPLE = Sample

        cls.parents = {
            cls.WRAPPER: None,
            cls.HISTORY: cls.WRAPPER,
            cls.TRACE: cls.HISTORY,
            cls.EPISODE: cls.TRACE,
            cls.STEP: cls.EPISODE,
            cls.SAMPLE: cls.STEP,
            None: cls.SAMPLE
        }

        cls.childs = {
            None: cls.WRAPPER,
            cls.WRAPPER: cls.HISTORY,
            cls.HISTORY: cls.TRACE,
            cls.TRACE: cls.EPISODE,
            cls.EPISODE: cls.STEP,
            cls.STEP: cls.SAMPLE,
            cls.SAMPLE: None,
        }

        cls.raw = None

    @classmethod
    def reinit(cls):
        cls.raw = cls.childs[None]()

    @classmethod
    def get_entity(cls, entity=None):
        if cls.raw is None: cls.reinit()

        if entity == type(cls.raw): return cls.raw
        try: return cls.get_entity(cls.parents[entity]).get(-1)
        except: return None

    @classmethod
    def set_entity(cls, entity):
        if cls.raw is None: cls.reinit()

        try: cls.get_entity(cls.parents[type(entity)]).set(entity)
        except:
            cls.set_entity(cls.parents[type(entity)]())
            cls.get_entity(cls.parents[type(entity)]).set(entity)
