import unittest
from time import time, sleep

from helpers import std_objects
from helpers.mocking.imitate_data_stream import DataSimulator


class TestML(unittest.TestCase):
    def setUp(self):
        self.data_simulator = DataSimulator(print_time=True, max_seconds=5, nusers=1)
        self.data_simulator.start()
        self.ml = self._create_ml()
        self.ml.start_fitting()

    def tearDown(self):
        self.ml.stop_fitting()
        self.data_simulator.stop()

    def test_can_predict_while_fitting(self):
        # can't start_fitting without users added, so add 1 user and generate only data from one user
        self.ml.add_user(user=0)
        start = time()
        self.ml.await_fit(5)  # awaiting so train can be started and if it contains bugs the errors will be raised
        recommends = self.ml.get_recommends(user=0)
        duration = time() - start
        self.assertLess(duration, 60 + 5)  # in seconds

    def test_add_user_then_can_give_recommends(self):
        user = 0
        self.ml.add_user(user=user)
        sleep(5)
        recommends = self.ml.get_recommends(user)
        # TODO: check recommends format in a more reliable and clean way
        print("final recommends in test", recommends)
        self.assertTrue(hasattr(recommends, "__iter__"))

    def _create_ml(self):
        return std_objects.get_ml()


if __name__ == '__main__':
    unittest.main()
