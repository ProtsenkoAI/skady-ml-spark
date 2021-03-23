import unittest
from time import time
import os

from helpers import tests_config, std_objects
from helpers.mocking.imitate_data_stream import DataSimulator

config = tests_config.TestsConfig()


class TestML(unittest.TestCase):
    def test_can_predict_while_fitting(self):
        data_simulator = DataSimulator(print_time=True, max_seconds=5, nusers=1)
        data_simulator.start()

        start = time()
        ml = self._create_ml()
        # can't start_fitting without users added, so add 1 user and generate only data from one user
        ml.add_user(user=0)
        ml.start_fitting()
        recommends = ml.get_recommends(user=0)
        duration = time() - start
        self.assertLess(duration, 20)  # in seconds

        ml.stop_fitting()
        data_simulator.stop()

    def test_add_user_then_can_give_recommends(self):
        ml = self._create_ml()
        user = 0
        ml.add_user(user=user)
        recommends = ml.get_recommends(user)
        # TODO: check recommends format in a more reliable and clean way
        self.assertTrue(hasattr(recommends, "__iter__"))

    def test_delete_user_then_can_not_predict(self):
        ml = self._create_ml()
        user = 0
        ml.add_user(user)
        ml.delete_user(user)
        with self.assertRaises(Exception):
            recommends = ml.get_recommends(user)
            print("delete recommends", recommends)

    def _create_ml(self):
        return std_objects.get_ml()


if __name__ == '__main__':
    unittest.main()
