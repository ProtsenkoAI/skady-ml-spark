import unittest
import numpy as np

from prod_manage.recommend_manager import RecommendManager
from model_level.assistant_builders import AssistantBuilder
from model_level.models.mf_with_bias import MFWithBiasModel
from ...helpers import std_objects

from ...helpers.tests_config import TestsConfig

config = TestsConfig()


class TestRecommendManager(unittest.TestCase):
    def setUp(self):
        self.user_colname = config.user_colname

    def test_fit(self):
        prod_manager = self._create_standard_manager()
        interacts = std_objects.get_interacts().copy()
        prod_manager.fit(interacts)

    def test_get_recommends(self):
        prod_manager = self._create_standard_manager()
        interacts = std_objects.get_interacts(nrows=200).copy()
        prod_manager.fit(interacts)
        some_users = interacts[self.user_colname].unique()[:3]
        recommends = prod_manager.get_recommends(some_users)
        self._validate_recommends_format(recommends, len(some_users))

    def test_update_with_new_users_get_recommends_for_them(self):
        prod_manager = self._create_standard_manager(nusers=1)
        interacts = std_objects.get_interacts(nrows=5).copy()
        prod_manager.fit(interacts)

        new_interacts = std_objects.get_interacts(nrows=300).copy()
        prod_manager.fit(new_interacts)
        new_users = new_interacts[self.user_colname].unique()
        recommends = prod_manager.get_recommends(new_users)
        self._validate_recommends_format(recommends, len(new_users))

    def _create_standard_manager(self, nusers=5):
        prod_manager = RecommendManager(std_objects.get_simple_trainer(), std_objects.get_recommender(),
                                        std_objects.get_standard_saver(), std_objects.get_assistant(nusers=nusers))
        return prod_manager

    def _validate_recommends_format(self, recommends, nb_users):
        self.assertEqual(len(recommends), nb_users)
        self.assertGreater(len(recommends[0]), 0, "recommendations for user shouldn't be empty list!")
        self.assertIsInstance(recommends[0][0], np.integer)
