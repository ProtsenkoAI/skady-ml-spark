import unittest
import pandas as pd

from recsys_pipeline.data import datasets
from ..helpers import tests_config
config = tests_config.TestsConfig()


class TestInteractsDataset(unittest.TestCase):
    def get_default_ratings(self):
        return pd.read_csv(config.interacts_path, nrows=20)

    def get_empty_ratings(self):
        return pd.DataFrame(columns=["user_id", "anime_id", "rating"])

    def create_dataset(self, ratings):
        return datasets.InteractDataset(ratings)


class TestInteractsDatasetCreation(TestInteractsDataset):
    def test_dataset_creation(self):
        ratings = self.get_default_ratings()
        dataset = self.create_dataset(ratings)

    def test_empty_dataset_creation(self):
        empty_ratings = self.get_empty_ratings()
        dataset = self.create_dataset(empty_ratings)

class TestInteractsDatasetIteration(TestInteractsDataset):
    def _test_iteration(self, dataset):
        for idx, sample in enumerate(dataset):
            with self.subTest(i=idx):
                self.assertTrue(len(sample) == 2)

    def test_dataset_iteration(self):
        """Test that we can iterate over dataset, and that every sample
        includes two elements: (features, labels)"""
        ratings = self.get_default_ratings()
        dataset = self.create_dataset(ratings)
        self._test_iteration(dataset)

    def test_empty_dataset_iteration(self):
        ratings = self.get_empty_ratings()
        dataset = self.create_dataset(ratings)
        self._test_iteration(dataset)

