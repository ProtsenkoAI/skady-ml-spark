import unittest
import pandas as pd
import torch

from recsys_pipeline.data import datasets_retrievers
from recsys_pipeline.managers import validators
from ..helpers import objects_creation, tests_config
config = tests_config.TestsConfig()


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.nusers = 100
        self.nitems = 100
        self.interacts_nrows = 100


        self.model = objects_creation.get_mf_model(self.nusers, self.nitems)
        self.interacts = objects_creation.get_interacts(self.interacts_nrows)
        all_item_ids = self.interacts["anime_id"].unique()
        self.dataloaders_retriever = objects_creation.get_datasets_retriever(self.interacts)
        self.preprocessor = objects_creation.get_preprocessor()
        self.validator = validators.Validator(self.model, self.dataloaders_retriever, all_item_ids,
                                              self.preprocessor)

    def test_evaluate(self):
        metric_val = self.validator.evaluate()
        self.assertIsInstance(metric_val, float)

    def test_one_user_evaluation(self):
        # maybe we shouldn't calculate metric for one user ?
        some_user = self.interacts[config.user_colname].unique()[0]
        user_interacts = self.interacts[self.interacts[config.user_colname] == some_user]
        all_item_ids = user_interacts["anime_id"].unique()

        users_retriever = objects_creation.get_datasets_retriever(user_interacts)
        dataset = next(iter(users_retriever))
        validator = validators.Validator(self.model, None, all_item_ids,
                                         self.preprocessor)

        score = validator._eval_user(dataset)
        self.assertIsInstance(score, pd.DataFrame)
