import unittest

from recsys_pipeline.trains_evals.evaluation import Validator
from ..helpers.objs_pool import ObjsPool
from ..helpers import std_objects, tests_config
objs_pool = ObjsPool()
config = tests_config.TestsConfig()


class TestValidator(unittest.TestCase):
    def test_evaluate_one_user_one_item_with_ndcg(self):
        assistant = std_objects.get_assistant()
        interacts = std_objects.get_interacts(nrows=1)
        assistant.update_with_interacts(interacts)
        validator = std_objects.get_validator()
        with self.assertWarns(UserWarning):
            eval_res = validator.evaluate(assistant, interacts)
        self.assertTrue(eval_res is None)  # can't calc ndcg of one user

    def test_evaluate_one_user_one_item_with_rmse(self):
        assistant = std_objects.get_assistant()
        interacts = std_objects.get_interacts(nrows=1)
        assistant.update_with_interacts(interacts)

        recommender = std_objects.get_recommender()
        validator_rmse = Validator(recommender, metric="rmse")
        eval_res = validator_rmse.evaluate(assistant, interacts)
        self.assertIsInstance(eval_res, float)

    def test_evaluate_many_users_many_items(self):
        assistant = std_objects.get_assistant()
        interacts = std_objects.get_interacts(nrows=20)
        assistant.update_with_interacts(interacts)
        validator = std_objects.get_validator()
        eval_res = validator.evaluate(assistant, interacts)
        self.assertIsInstance(eval_res, float)
