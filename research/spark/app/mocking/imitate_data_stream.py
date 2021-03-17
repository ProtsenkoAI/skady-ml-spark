import time
import os
from random import uniform

from util import read_config
from .generators import CsvDirGenerator

conf = read_config()
data_dir = os.path.join(conf["base_path"], conf["fit_stream"]["relative_path"])
data_generator = CsvDirGenerator(data_dir)


def _sleep_some_time(min_seconds=1, max_seconds=10):
        time_to_sleep = uniform(min_seconds, max_seconds)
        time.sleep(time_to_sleep)


def run(max_seconds=2, print_time=False):
    while True:
        data_generator.create_push()
        if print_time:
            print("pushed", time.time())
        _sleep_some_time(max_seconds=max_seconds)
