import os
import sys
import time
import pprint
import signal
import logging
import threading
import multiprocessing

from datetime import datetime

from gazebo_sim.utils.ArgsConfig import Args
from gazebo_sim.utils.Exchanging import Exchanger


class Logger():
    def __new__(cls):
        try: return cls.__instance__
        except:
            cls.initialize_once()
            cls.__instance__ = super().__new__(cls)
            return cls.__instance__

    @classmethod
    def initialize_once(cls):
        cls.DEBUG = logging.DEBUG # 10
        cls.INFO  = logging.INFO  # 20
        cls.WARN  = logging.WARN  # 30
        cls.ERROR = logging.ERROR # 40

        fallback = cls.ERROR

        # read from argparser and set for all existing loggers this level
        logging.basicConfig(level=fallback, force=True)

        # for name, logger in logging.Logger.manager.loggerDict.items():
        #     if name.find('ray') != -1 or name.find('rllib') != -1:
        #         try: logger.setLevel(fallback)
        #         except: pass

        cls.logging_mode = Args().logging_mode
        cls.console_level = cls.convert(Args().console_level)
        cls.file_level = cls.convert(Args().file_level)

        if cls.console_level is not None:
            log_path = sys.stdout
            # create console handler and formatter
            console_handler = logging.StreamHandler(log_path)
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            # create console logger and add handler
            cls.console_logger = logging.Logger(f'console_logger ({Args().ENTRY})')
            cls.console_logger.setLevel(cls.console_level)
            cls.console_logger.addHandler(console_handler)
        else:
            cls.console_logger = None

        if cls.file_level is not None:
            log_path = os.path.join(Exchanger().path_register['log'], f'{Args().ENTRY}.log')
            # create file handler and formatter
            file_handler = logging.FileHandler(log_path)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            # create file logger and add handler
            cls.file_logger = logging.Logger(f'file_logger ({Args().ENTRY})')
            cls.file_logger.setLevel(cls.file_level)
            cls.file_logger.addHandler(file_handler)
        else:
            cls.file_logger = None

        # from logging.handlers import QueueHandler, QueueListener
        # QueueHandler(cls.queue)
        # QueueListener(cls.queue, console_handler, file_handler)

        if cls.logging_mode == 'sync': cls.immediately = True
        elif cls.logging_mode == 'async': cls.immediately = False
        else: cls.immediately = None

        if not cls.immediately: cls.init_async()

    @classmethod
    def init_async(cls):
        # cls.thread = threading.main_thread()
        cls.queue = multiprocessing.Queue()

        cls.resume = multiprocessing.Event()
        cls.resume.clear()

        # cls.finish = multiprocessing.Event()
        # cls.finish.clear()

        cls.process = multiprocessing.Process(target=cls.run_async)
        cls.process.start()

        # for status in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
        #     signal.signal(status, lambda signum, frame: cls.finish.set())
        #     signal.signal(status, lambda signum, frame: cls.del_async())

    @classmethod
    def callback_async(cls, args):
        # cls.queue.put(args)
        # if not cls.process.is_alive():
        #     cls.process.start()
        cls.queue.put(args)
        if not cls.resume.is_set(): cls.resume.set()

    @classmethod
    def run_async(cls):
        # while not cls.queue.empty():
        #     func, args = cls.queue.get()
        #     func(*args)
        # while cls.thread.is_alive() or not cls.queue.empty():
        #     if cls.queue.empty():
        #         cls.resume.clear()
        #         cls.resume.wait()
        #     func, args = cls.queue.get()
        #     func(*args)
        # while not (cls.finish.is_set() and cls.queue.empty()):
        #     if cls.queue.empty():
        #         cls.resume.clear()
        #         cls.resume.wait()
        #     func, args = cls.queue.get()
        #     func(*args)
        # while not (cls.finish.is_set() and cls.queue.empty()):
        #     func, args = cls.queue.get()
        #     func(*args)
        while True:
            if cls.queue.empty():
                cls.resume.clear()
                cls.resume.wait()
            args = cls.queue.get()
            cls.perform(*args)

    @classmethod
    def del_async(cls):
        if not cls.immediately:
            # cls.finish.set()
            if not cls.resume.is_set():
                cls.resume.set()
            while not cls.queue.empty():
                time.sleep(1)
            cls.process.terminate()

    @classmethod
    def convert(cls, verbosity):
        if verbosity == 0: return None
        elif verbosity == 1: return cls.ERROR
        elif verbosity == 2: return cls.WARN
        elif verbosity == 3: return cls.INFO
        elif verbosity == 4: return cls.DEBUG

    @classmethod
    def debug(cls, msg, formatted=False):
        cls.output(cls.DEBUG, msg, formatted)

    @classmethod
    def info(cls, msg, formatted=False):
        cls.output(cls.INFO, msg, formatted)

    @classmethod
    def warn(cls, msg, formatted=False):
        cls.output(cls.WARN, msg, formatted)

    @classmethod
    def error(cls, msg, formatted=False):
        cls.output(cls.ERROR, msg, formatted)

    @classmethod
    def output(cls, level, msg, formatted=False):
        # NOTE: track discrepancy between created and logged in async mode
        msg = f'[{datetime.now()}] {msg}'

        if not cls.void(level):
            if cls.immediately: cls.perform(level, msg, formatted)
            else: cls.callback_async((level, msg, formatted))

    @classmethod
    def void(cls, level):
        if cls.logging_mode is None: return True
        else:
            if cls.console_level is None:
                try: return level < cls.file_level
                except: return True
            if cls.file_level is None:
                try: return level < cls.console_level
                except: return True
            return level < cls.console_level and level < cls.file_level

    @classmethod
    def perform(cls, level, msg, formatted):
        if formatted: msg = pprint.pformat(msg)

        if cls.console_logger is not None: cls.console_logger.log(level, msg)
        if cls.file_logger is not None: cls.file_logger.log(level, msg)
