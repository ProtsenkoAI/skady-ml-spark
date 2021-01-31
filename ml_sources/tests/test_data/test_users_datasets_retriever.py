import unittest
import pandas as pd
from torch.utils import data as torch_data

from recsys_pipeline.data import datasets_retrievers, loader_build
from ..helpers import tests_config
config = tests_config.TestsConfig()


class TestUsersDatasetsRetriever(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.nrows = 400
        self.users_datasets_retriever = self._create_standard_retriever()

    def test_retrieved_sample_shape_and_dtypes(self):
        sample = next(iter(self.users_datasets_retriever))
        data_corr_type = isinstance(sample, torch_data.DataLoader)
        self.assertTrue(data_corr_type, "Sample elements have wrong types")

    def test_retrieved_samples_are_different(self):
        retriever_iter = iter(self.users_datasets_retriever)
        sample1 = next(retriever_iter)
        sample2 = next(retriever_iter)
        self.assertFalse(sample1 == sample2)

    def _create_standard_retriever(self):
        loader_builder = loader_build.StandardLoaderBuilder(self.batch_size)
        interacts = pd.read_csv(config.interacts_path, nrows=self.nrows)
        return datasets_retrievers.UsersDatasetsRetriever(interacts, loader_builder)
