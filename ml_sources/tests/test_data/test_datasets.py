import unittest
from torch.utils import data as torch_data

from data.datasets import InteractDataset
from ..helpers.objs_pool import ObjsPool
from ..helpers import std_objects, tests_config
objs_pool = ObjsPool()
config = tests_config.TestsConfig()


class TestInteractDataset(unittest.TestCase):
    def test_iterate(self):
        interacts = std_objects.get_interacts()
        dataset = InteractDataset(interacts)
        idx = 0
        for elem in dataset:
            idx += 1
            if idx == 5:
                break

    def test_get_len(self):
        interacts = std_objects.get_interacts(20)
        dataset = InteractDataset(interacts)
        self.assertEqual(len(dataset), 20)
