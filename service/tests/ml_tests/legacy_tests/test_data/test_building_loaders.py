import unittest
from torch.utils import data as torch_data

from data import StandardLoaderBuilder
from helpers import tests_config, std_objects

config = tests_config.TestsConfig()


class TestStandardLoaderBuilder(unittest.TestCase):
    def test_build(self):
        builder = StandardLoaderBuilder(batch_size=44)
        interacts = std_objects.get_interacts(nrows=14)
        loader = builder.build(interacts)
        self.assertIsInstance(loader, torch_data.DataLoader)

