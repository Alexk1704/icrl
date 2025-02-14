import os
import sys
import ray
import time
import itertools
import functools

from datetime import datetime

from ray import tune
from ray.rllib.models import ModelCatalog


from rllib_gazebo.utils.Evaluation import *
from rllib_gazebo.utils.ArgsConfig import LearnerArgs
from rllib_gazebo.utils.Exchanging import Exchanger
from rllib_gazebo.utils.Logging import Logger
from rllib_gazebo.utils.Caching import Cache

from rllib_gazebo.envs.LineFollowing import LineFollowingWrapper

import rllib_gazebo.algorithms.dqn as DQN
import rllib_gazebo.algorithms.qgmm as QGMM
import rllib_gazebo.algorithms.pg as PG

from cl_replay.architecture.ar.callback import Log_Protos


class RLConfig():
    def __init__(self) -> None:
        for name, value in vars(LearnerArgs().args).items():
            setattr(self, name, value)

        self.envs = {
            'LF': LineFollowingWrapper,
        }

        self.algos = {
            'QGMM': QGMM,
            'DQN': DQN,
            'PG': PG,
        }


class RLTrainer():
    def __init__(self, config:RLConfig) -> None:
        self.config = config

        env = self.config.envs[self.config.environment]
        algo = self.config.algos[self.config.backend]

        # start Ray and register custom environment
        env_name = 'CustomEnvironment-v0'
        ray.init(ignore_reinit_error=True, logging_level=Logger().ERROR)
        tune.register_env(env_name, env)

        # configure environment and create agent
        self.generate_benchmark_agent(self.config, algo, env_name)
        self.examine_policy_model()
        
        self.latest_checkpoint_file = None

    def generate_benchmark_agent(self, args, algo, env):
        model_name = 'custom_tf2_model'
        ModelCatalog.register_custom_model(model_name, algo.CustomQTF2Model)

        self.algo_config = algo.CustomQTF2AlgoConfig() # create policy specific config
        self.algo_config.model.update({
            'custom_model': model_name,
            'fcnet_hiddens': args.fcnet_hiddens,
            'post_fcnet_hiddens': args.post_fcnet_hiddens,
            'fcnet_activation': args.fcnet_activation,
            'post_fcnet_activation': args.post_fcnet_activation,
            'no_final_linear': args.no_final_linear,
            'vf_share_layers': args.vf_share_layers,
            'free_log_std': args.free_log_std
        })

        self.algo_config = self.override_config(self.algo_config, args) # update with global settings
        self.agent = self.algo_config.build(env=env); self.agent_stopped = False

        Exchanger().store(self.algo_config.serialize(), 'agent.json', Exchanger().path_register['config'])
        Logger().debug(f'Build {algo} agent')

    def override_config(self, config, args):
        config.resources(
            num_gpus=0 if args.cpu_only else 1,
        ).framework(
            framework='tf2',
            eager_tracing=False,
        ).environment(
            env=None,
            env_config={},
            env_task_fn=None,
            disable_env_checking=True,
        ).rollouts(
            rollout_fragment_length='auto',
            batch_mode='truncate_episodes',
        ).training(
            gamma=args.gamma,
            lr=args.lr,
            train_batch_size=args.train_batch_size,
        ).evaluation(
            evaluation_duration=1,
            evaluation_duration_unit=args.evaluation_duration_unit,
            evaluation_config={'explore': False},
        ).reporting(
            min_time_s_per_iteration=None,
            min_train_timesteps_per_iteration=None,
            min_sample_timesteps_per_iteration=None,
        ).debugging(
            log_level=args.verbose,
        )

        return config

    def examine_policy_model(self):
        policy = self.agent.get_policy()
        model = policy.model

        Logger().debug(f'examine model {model} of policy {policy}')

        try:
            try: summary = model.logits_and_value_model.summary()
            except: summary = model.base_model.summary()

            Exchanger().store(summary, 'model.json', Exchanger().path_register['info'])
        except:
            Logger().error('Examination not supported!')

    def train(self, track):
        Logger().info(f'Begin training on {track}')
        # NOTE: see line_following_wrapper.py: LineFollowingWrapper.switch(mode, track)
        env = self.agent.workers.local_worker().env
        env.switch('train', track)
        
        if self.agent_stopped: 
            self.agent = self.algo_config.build() # restore agent after close
    
        if self.config.checkpointing and self.latest_checkpoint_file:
            self.agent.restore(self.latest_checkpoint_file) # restore from check-point (load state)

        # bm = self.agent.get_policy().model.base_model
        # tm = self.agent.get_policy().target_model.base_model
        # qvh = self.agent.get_policy().model.q_value_head
        # svh = self.agent.get_policy().model.state_value_head
        
        # print(bm)
        # print(tm)
        # print(qvh)
        # print(svh)

        # import numpy as np 
        # print("!!! AFTER RESTORE !!!")
        # print("!!! BM:")
        # for layer in bm.layers:
        #     print(layer.name, ":")
        #     for var in layer.variables:
        #         print("\t", var.name, "\t", np.mean(var.numpy()))
        # print("!!! TM:")    
        # for layer in tm.layers:
        #     print(layer.name, ":")
        #     for var in layer.variables:
        #         print("\t", var.name, "\t", np.mean(var.numpy()))
        # print("!!! QVH:")        
        # for layer in qvh.layers:
        #     print(layer.name, ":")
        #     for var in layer.variables:
        #         print("\t", var.name, "\t", np.mean(var.numpy()))

        self.agent.before_subtask('train')
        
        if self.config.training_duration_unit == 'timesteps':
            frequency = 100
            count = max(1, self.config.training_duration // frequency)
            for i in range(self.config.training_duration):
                if self.config.debug or i % count == 0:
                    Logger().info(f'{i + 1}/{self.config.training_duration} {self.config.training_duration_unit}')
                _ = self.agent.train(track)
                
        self.agent.after_subtask('train')

        if self.config.checkpointing: 
            self.latest_checkpoint_file = self.agent.save(Exchanger().path_register['checkpoint'])
        self.agent.stop(); self.agent_stopped = True
        
        Logger().info(f'Finish training on {track}')

    def evaluate(self, track):
        Logger().info(f'Begin evaluate on {track}')
        # NOTE: see line_following_wrapper.py: LineFollowingWrapper.switch(mode, track)
        env = self.agent.workers.local_worker().env
        env.switch('eval', track)
        
        if self.agent_stopped: 
            self.agent = self.algo_config.build()

        if self.config.checkpointing and self.latest_checkpoint_file:
            self.agent.restore(self.latest_checkpoint_file)

        self.agent.before_subtask('eval')
        
        if self.config.evaluation_duration_unit == 'timesteps':
            frequency = 100
            count = max(1, self.config.evaluation_duration // frequency)
            for i in range(self.config.evaluation_duration):
                if self.config.debug or i % count == 0:
                    Logger().info(f'{i + 1}/{self.config.evaluation_duration} {self.config.evaluation_duration_unit}')
                _ = self.agent.evaluate(track)

        self.agent.after_subtask('eval')
        self.agent.stop(); self.agent_stopped = True
        
        Logger().info(f'Finish evaluate on {track}')
        
    def store(self, entity, context):
        entry = Cache().object_registry['counters'][0]
        entry[-1].append(self.agent.workers.local_worker().env.data.counters['tick'])
        Cache().update_object('counters', entry)

        # ---

        Cache().update_object('samples', Evaluator().raw) # HACK

        Exchanger().bulk_store(trigger=entity)

        if entity == self.config.report_level:
            cbs = []
            if context == self.train:
                try: cbs.extend(self.agent.get_policy().model.train_callbacks)
                except: pass
            if context == self.evaluate:
                try: cbs.extend(self.agent.get_policy().model.evaluate_callbacks)
                except: pass

            for cb in cbs:
                if isinstance(cb, Log_Protos): cb.save()
                # further storage cbs with their conditions

    def terminate(self):
        Logger().warn('terminating the agent')
        self.agent.workers.local_worker().env.destroy()
        self.agent.stop()


class RLIterator():
    def __init__(self, config:RLConfig, train_func:RLTrainer.train, eval_func:RLTrainer.evaluate) -> None:
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
    config = RLConfig()
    trainer = RLTrainer(config)
    iterator = RLIterator(config, trainer.train, trainer.evaluate)

    iterator.store()

    # use a while loop in combination with next()
    for context, subtask in iterator:
        start_time = time.time_ns()
        context(subtask)
        trainer.store('switch', context)
        end_time = time.time_ns()
        elapsed_time = (end_time - start_time) / 1e+09
        Logger().info(f'Elapsed time for subtask: {elapsed_time}')

    trainer.terminate()
    Logger().info(f'Finish execution at: {datetime.now()}')
    Exchanger().store(Cache().object_registry['counters'][0], 'counters.json', Exchanger().path_register['info'])


if __name__== '__main__':
    # explicitly init exchanger and logger
    Exchanger()
    Logger()

    try:
        main()
    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPT')
    finally:
        Logger().del_async()
        # sys.exit(0)
        # os._exit(0)
