import unittest

from data.datasets import InteractDataset

from helpers import tests_config, std_objects

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
