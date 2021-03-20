import unittest
from time import time

from helpers import tests_config, std_objects

config = tests_config.TestsConfig()


class TestML(unittest.TestCase):
    def test_can_predict_while_fitting(self):
        start = time()
        ml = self._create_ml()
        ml.start_fitting()
        user = 1
        recommends = ml.get_recommends(user)
        duration = time() - start
        self.assertLess(duration, 5)
        ml.stop_fitting()

    def test_predict(self):
        raise NotImplementedError

    def test_add_user_then_can_predict(self):
        raise NotImplementedError

    def test_delete_user_then_can_not_predict(self):
        raise NotImplementedError

    def _create_ml(self):
        return std_objects.get_ml()