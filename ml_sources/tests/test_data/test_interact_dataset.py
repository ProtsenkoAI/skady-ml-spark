import unittest
import pandas as pd

from recsys_pipeline.data import datasets
from ..helpers import tests_config, objects_creation
config = tests_config.TestsConfig()


class TestInteractsDataset(unittest.TestCase):
    def setUp(self):
        self.standard_nrows = 20

    def test_dataset_iteration(self):
        """Test that we can iterate over dataset, and that every sample
        includes two elements: (features, labels)"""
        ratings = self._get_default_interacts()
        dataset = self._create_dataset(ratings)
        self._test_iteration(dataset)

    def test_empty_dataset_iteration(self):
        ratings = self._get_empty_interacts()
        dataset = self._create_dataset(ratings)
        self._test_iteration(dataset)

    def _test_iteration(self, dataset):
        for idx, sample in enumerate(dataset):
            with self.subTest(i=idx):
                self.assertTrue(len(sample) == 2) # features and labels

    def _get_default_interacts(self):
        return objects_creation.get_interacts(self.standard_nrows)

    def _get_empty_interacts(self):
        return objects_creation.get_interacts(nrows=0)

    def _create_dataset(self, ratings):
        return datasets.InteractDataset(ratings)
