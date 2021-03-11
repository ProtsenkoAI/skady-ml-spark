import time
from random import uniform

from util import read_config
from .generators import TextDirGenerator

conf = read_config()
data_dir = conf["data_stream"]["path"]
data_generator = TextDirGenerator(data_dir)


def _sleep_some_time(min_seconds=1, max_seconds=10):
        time_to_sleep = uniform(min_seconds, max_seconds)
        time.sleep(time_to_sleep)


def run():
    while True:
        data_generator.create_push()
        _sleep_some_time()