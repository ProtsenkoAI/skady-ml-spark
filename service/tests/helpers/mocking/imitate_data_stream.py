import time
import os
from random import uniform
import shutil
import threading

from ml_util import read_config
from .generators import CsvDirGenerator

conf = read_config()
data_dir = os.path.join(conf["common_params"]["paths"]["base_path"], conf["obtainer_params"]["fit_stream"]["relative_path"])
shutil.rmtree(data_dir, ignore_errors=True)


def _sleep_some_time(min_seconds=1, max_seconds=10):
    time_to_sleep = uniform(min_seconds, max_seconds)
    time.sleep(time_to_sleep)


class DataSimulator:
    # TODO: maybe add "with" support?
    def __init__(self, print_time, max_seconds, nusers):
        self.data_generation_process = GenDataThread(print_time, max_seconds, nusers)

    def start(self):
        print("creating dir", data_dir)
        os.makedirs(data_dir, exist_ok=True)
        self.data_generation_process.start()

    def stop(self):
        """deletes generated data"""
        print("stopping data generation")
        self.data_generation_process.stop()
        shutil.rmtree(data_dir)


class GenDataThread(threading.Thread):
    def __init__(self, print_time, max_seconds, nusers, *args, **kwargs):
        self.print_time = print_time
        self.max_seconds = max_seconds
        self.data_generator = CsvDirGenerator(data_dir, nusers=nusers)
        super(GenDataThread, self).__init__(name="GenDataThread", *args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        while True:
            if self._stop_event.is_set():
                break
            self.data_generator.create_push()
            if self.print_time:
                print("pushed", time.time())
            _sleep_some_time(max_seconds=self.max_seconds)
