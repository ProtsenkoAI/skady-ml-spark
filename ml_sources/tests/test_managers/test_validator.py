import unittest
import pandas as pd
import torch

from recsys_pipeline.data import users_datasets_retriever
from recsys_pipeline.managers import validators
from . import helpers

class TestValidator(unittest.TestCase):
    def setUp(self):
        self.nusers = 100
        self.nitems = 100
        self.interacts_nrows = 100

        self.model = helpers.get_model(self.nusers, self.nitems)
        self.interacts = helpers.load_interacts(self.interacts_nrows)
        all_item_ids = self.interacts["anime_id"].unique()
        # self.dataloader = helpers.create_dataloader(self.interacts)
        self.dataloaders_retriever = users_datasets_retriever.UsersDatasetsRetriever(self.interacts)
        self.preprocessor = helpers.get_preprocessor()
        self.validator = validators.Validator(self.model, self.dataloaders_retriever, all_item_ids,
                                        self.preprocessor)

    def test_evaluate(self):
        metric_val = self.validator.evaluate()
        self.assertIsInstance(metric_val, float)

    def test_one_user_evaluation(self):
        some_user = self.interacts["user_id"].unique()[0]
        user_interacts = self.interacts[self.interacts["user_id"] == some_user]
        all_item_ids = user_interacts["anime_id"].unique()

        user, dataset = next(iter(users_datasets_retriever.UsersDatasetsRetriever(user_interacts)))
        validator = validators.Validator(self.model, dataset, all_item_ids, 
                                            self.preprocessor)

        score = validator._eval_user(user, dataset)
        self.assertIsInstance(score, pd.DataFrame)
