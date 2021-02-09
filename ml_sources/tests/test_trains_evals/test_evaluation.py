import unittest

from ..helpers.objs_pool import ObjsPool
from ..helpers import std_objects, tests_config
objs_pool = ObjsPool()
config = tests_config.TestsConfig()


# class TestValidator(unittest.TestCase):
#     def test_evaluate_one_user_one_item(self):
#         assistant = std_objects.get_assistant()
#         interacts = std_objects.get_interacts(nrows=1)
#         validator = std_objects.get_validator()
#         eval_res = validator.evaluate(assistant, interacts)
#         self.assertIsInstance(eval_res, int)
#
#     def test_evaluate_many_users_many_items(self):
#         assistant = std_objects.get_assistant()
#         interacts = std_objects.get_interacts(nrows=20)
#         validator = std_objects.get_validator()
#         eval_res = validator.evaluate(assistant, interacts)
#         self.assertIsInstance(eval_res, int)