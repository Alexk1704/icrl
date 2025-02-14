import os
import time
import numpy as np

from datetime import datetime

from gazebo_sim.utils.ArgsConfig import Args
from gazebo_sim.utils.Caching import Cache
from gazebo_sim.utils.Evaluation import *
from gazebo_sim.utils.Exchanging import Exchanger
from gazebo_sim.utils.Logging import Logger

from gazebo_sim.learner.LFLearner import LFLearner
from gazebo_sim.simulation.LineFollowing import LineFollowingWrapper

from cl_replay.architecture.ar.callback import Log_Protos


class RLAgent():

    def __init__(self, config):
        self.config = config
        
        self.environment = LineFollowingWrapper(
            step_duration_nsec = config.step_duration_nsec
        )
        
        self.learner = LFLearner(
            obs_space=self.environment.observation_space,
            config=config
        )

    def train(self, track):
        Logger().info(f'Begin training on {track}')
        
        self.environment.switch('train', track)
        
        self.learner.before_train()

        if self.config.training_duration_unit == 'timesteps':
            frequency = 100
            count = max(1, self.config.training_duration // frequency)
            
            done = True; last_obs = None
            for i in range(self.config.training_duration):
                if self.config.debug or i % count == 0: 
                    Logger().info(f'{i + 1}/{self.config.training_duration} timesteps')
                done, last_obs = do_step(self.learner, self.environment, done=done, last_obs=last_obs, train=True)
        # -----
        if self.config.training_duration_unit == 'episodes':
            frequency = 100
            count = max(1, self.config.training_duration // frequency)
            
            for i in range(self.config.training_duration):
                if self.config.debug or i % count == 0: 
                    Logger().info(f'{i + 1}/{self.config.training_duration} episodes')
                do_episode(self.learner, self.environment, train=True)
        
        self.learner.after_train()

        Logger().info(f'Finish training on {track}')

    def evaluate(self, track):
        Logger().info(f'Begin evaluate on {track}')

        self.environment.switch('eval', track) 
        
        self.learner.before_evaluate()
        if self.config.evaluation_duration_unit == 'timesteps':
            frequency = 1000
            count = max(1, self.config.training_duration // frequency)
            
            done = True; last_obs = None
            for i in range(self.config.evaluation_duration):
                if self.config.debug or i % count == 0: 
                    Logger().info(f'{i + 1}/{self.config.evaluation_duration} timesteps')
                done, last_obs = do_step(self.learner, self.environment, done=done, last_obs=last_obs, train=False)
        # -----
        if self.config.evaluation_duration_unit == 'episodes':
            frequency = 100
            count = max(1, self.config.evaluation_duration // frequency)
            
            for i in range(self.config.evaluation_duration):
                if self.config.debug or i % count == 0:
                    Logger().info(f'{i + 1}/{self.config.evaluation_duration} episodes')
                do_episode(self.learner, self.environment, train=False)
        self.learner.after_evaluate()
        
        Logger().info(f'Finish evaluate on {track}')

    def store(self, entity, context):
        entry = Cache().object_registry['counters'][0]
        entry[-1].append(self.environment.data.counters['tick'])
        Cache().update_object('counters', entry)

        Cache().update_object('samples', Evaluator().raw) # HACK: call .raw to make sure object data is saved!

        Exchanger().bulk_store(trigger=entity) # NOTE: entity (episode/task) determines when cache is stored!
        
        # NOTE: no callbacks needed rn; comment in and trigger callbacks store function manually
        # if entity == self.config.report_level:
        #     cbs = []
        #     if context == self.train:
        #         try: cbs.extend(self.learner.model.train_callbacks)
        #         except: pass
        #     if context == self.evaluate:
        #         try: cbs.extend(self.learner.model.evaluate_callbacks)
        #         except: pass
        
        #     for cb in cbs:
        #         if isinstance(cb, Log_Protos): cb.save()
        #         # further storage cbs with their conditions

    def close(self):
        self.environment.close()

def do_episode(learner, environment, train=False):
    truncated = False; terminated = False
    observation_tm1, _ = environment.reset()
    
    while not terminated and not truncated:
        # NOTE: force agent to perform a deterministic action for the first 2 steps!
        # if environment.step_count < 2:  
        #     action_t, randomly_chosen = learner.choose_action(None, force=0)  # FORCE STOP
        # else:
        action_t, randomly_chosen = learner.choose_action(observation_tm1)

        observation_t, reward_t, terminated, truncated, _ = environment.step(action_t, randomly_chosen)
        # print(reward_t, terminated, truncated, randomly_chosen, learner.epsilon)
        
        if train:
            if environment.step_count > 2:
                learner.store_transition(
                    observation_tm1,
                    action_t,
                    reward_t,
                    observation_t,
                    terminated or truncated,
                )
            if environment.step_count < 2: continue  # NOTE: skip training for first 2 steps due to incomplete obs.

        observation_tm1 = observation_t

def do_step(learner, environment, done, last_obs, train=False):
    truncated = False; terminated = False
    
    if done:
        observation_tm1, _ = environment.reset()
    else:
        observation_tm1 = last_obs
        
    # NOTE: force agent to perform a deterministic action for the first 2 steps!
    # if environment.step_count < 2:  
    #     action_t, randomly_chosen = learner.choose_action(None, force=0)  # FORCE STOP
    # else:
    action_t, randomly_chosen = learner.choose_action(observation_tm1)
    # action_t, randomly_chosen = learner.choose_action(None, force=0)

    observation_t, reward_t, terminated, truncated, _ = environment.step(action_t, randomly_chosen)
    # print(reward_t, terminated, truncated, randomly_chosen, learner.epsilon)

    if train:
        if environment.step_count > 2:
            learner.store_transition(
                observation_tm1,
                action_t,
                reward_t,
                observation_t,
                terminated or truncated,
            )
            learner.learn() # NOTE: skip storing & training for first 2 steps due to incomplete obs.

    return (terminated or truncated), observation_t


class RLIterator():
    def __init__(self, config, train_func:RLAgent.train, eval_func:RLAgent.evaluate) -> None:
        self.config = config

        self._pipeline = []
        self.build(train_func, eval_func)

    def __iter__(self):
        self._counter = -1
        return self

    def __next__(self):
        try:
            self._counter += 1
            return self._pipeline[self._counter]
        except:
            raise StopIteration

    def build(self, train_func, eval_func):
        # hack some dirty solution via itertools interchangeable

        temp = {
            'train': {'func': train_func, 'subtasks': self.config.train_subtasks, 'swap': self.config.train_swap, 'cnt': 0},
            'eval': {'func': eval_func, 'subtasks': self.config.eval_subtasks, 'swap': self.config.eval_swap, 'cnt': 0},
        }

        def extend_pipeline(context):
            func, subtasks, swap, cnt = temp[context].values()

            if len(subtasks) == 0 or swap == 0: return

            if swap == -1: swap = len(subtasks)
            swap = min(swap, len(subtasks))

            temp[context]['cnt'] += swap
            self._pipeline.extend([(func, subtasks[(cnt + i) % len(subtasks)]) for i in range(swap)])

        # rename to interchanging and sequential?!
        if self.config.context_change == 'alternately':
            if self.config.begin_with == 'train': extend_pipeline('train')
            while temp['train']['cnt'] // len(temp['train']['subtasks']) < self.config.task_repetition:
                extend_pipeline('eval') ; extend_pipeline('train')
            if self.config.end_with == 'eval': extend_pipeline('eval')

        if self.config.context_change == 'completely':
            temp['eval']['swap'] = -1
            temp['train']['swap'] = -1
            if self.config.begin_with == 'train': extend_pipeline('train')
            while temp['train']['cnt'] // len(temp['train']['subtasks']) < self.config.task_repetition:
                extend_pipeline('eval') ; extend_pipeline('train')
            if self.config.end_with == 'eval': extend_pipeline('eval')

    def pipeline(self):
        return [{'mode': str(func.__name__), 'name': str(subtask)} for func, subtask in self._pipeline]

    def store(self):
        Exchanger().store(self.pipeline(), 'tasks.json', Exchanger().path_register['info'])
        

def main():
    Logger().info(f'Begin execution at: {datetime.now()}')
    
    config = Args().args
    agent = RLAgent(config) 
    iterator = RLIterator(config, agent.train, agent.evaluate)
    iterator.store()

    for context, subtask in iterator:
        start_time = time.time_ns()
        context(subtask)
        agent.store('switch', context)
        end_time = time.time_ns()
        elapsed_time = (end_time - start_time) / 1e+09
        Logger().info(f'Elapsed time for subtask: {elapsed_time}')

    Logger().info(f'Finish execution at: {datetime.now()}')
    agent.close()
    Exchanger().store(Cache().object_registry['counters'][0], 'counters.json', Exchanger().path_register['info'])

if __name__ == "__main__":
    # explicitly init exchanger and logger
    Args()
    Logger()
    Exchanger()

    try:
        main()
    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPT')
    finally:
        Logger().del_async()
        # sys.exit(0)
        # os._exit(0)
