import unittest
from torch.utils import data as torch_data

from data.building_loaders import StandardLoaderBuilder
from ..helpers import std_objects, tests_config
config = tests_config.TestsConfig()


class TestStandardLoaderBuilder(unittest.TestCase):
    def test_build(self):
        builder = StandardLoaderBuilder(batch_size=44)
        interacts = std_objects.get_interacts(nrows=14)
        loader = builder.build(interacts)
        self.assertIsInstance(loader, torch_data.DataLoader)

