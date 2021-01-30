import unittest
import pandas as pd
from torch.utils import data as torch_data

from recsys_pipeline.data import users_datasets_retriever
from ..helpers import tests_config
config = tests_config.TestsConfig()


class TestUsersDatasetsRetriever(unittest.TestCase):
    def test_create_retriever(self):
        ratings = self._get_default_ratings()
        self._create_retriever(ratings)

    def test_retrieved_sample_shape_and_dtypes(self):
        ratings = self._get_default_ratings()
        retriever = self._create_retriever(ratings)

        sample = next(iter(retriever))
        user_id, user_data = sample

        id_corr_type = isinstance(user_id, int)
        data_corr_type = isinstance(user_data, pd.DataFrame)
        self.assertTrue(id_corr_type and data_corr_type, "Sample elements have "
                                                          "wrong types")

    def test_retrieved_samples_are_different(self):
        ratings = self._get_default_ratings()
        retriever = self._create_retriever(ratings)
        retriever_iter = iter(retriever)
        sample1 = next(retriever_iter)
        sample2 = next(retriever_iter)
        self.assertFalse(sample1 == sample2)
                                                          
    def _create_retriever(self, ratings):
        return users_datasets_retriever.UsersDatasetsRetriever(ratings)
    
    def _get_default_ratings(self):
        return pd.read_csv(config.interacts_path, nrows=400)