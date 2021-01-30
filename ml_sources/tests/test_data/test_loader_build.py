# TODO
import unittest
import pandas as pd

from recsys_pipeline.data import loader_build, datasets
from ..helpers import tests_config
config = tests_config.TestsConfig()


class TestStandardLoaderBuilder(unittest.TestCase):
    def test_init_and_call(self):
        interacts = pd.read_csv(config.interacts_path, nrows=100)
        dataset = datasets.InteractDataset(interacts)
        builder = loader_build.StandardLoaderBuilder(batch_size=12)
        loader = builder(dataset)
