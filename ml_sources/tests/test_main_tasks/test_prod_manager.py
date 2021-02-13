import unittest

from high_level_managing.prod_manager import ProdManager
from model_level.assistant_builders import AssistantBuilder
from model_level.models.mf_with_bias import MFWithBiasModel
from ..helpers import std_objects
from ..helpers.objs_pool import ObjsPool
from ..helpers.tests_config import TestsConfig
objs_pool = ObjsPool()
config = TestsConfig()


class TestTrainProdManager(unittest.TestCase):
    def setUp(self):
        self.user_colname = config.user_colname

    def test_add_interacts(self):
        prod_manager = self._create_standard_manager()
        interacts = std_objects.get_interacts().copy()
        prod_manager.add_interacts(interacts)

    def test_fit(self):
        prod_manager = self._create_standard_manager()
        interacts = std_objects.get_interacts().copy()
        prod_manager.add_interacts(interacts)
        prod_manager.fit(nepochs=2)

    # TODO
    # def test_get_recommends(self):
    #     prod_manager = self._create_standard_manager()
    #     interacts = std_objects.get_interacts().copy()
    #     prod_manager.add_interacts(interacts)
    #     some_users = interacts[self.user_colname].unique()[:3]
    #     recommends = prod_manager.get_recommends(some_users)
    #     self._validate_recommends_format(recommends, len(some_users))

    def test_save_load(self):
        prod_manager = self._create_standard_manager()
        prod_manager.save()

    def test_build_new_assistant(self):
        assist_builder = AssistantBuilder(MFWithBiasModel, nusers=10, nitems=10, hidden_size=5)
        prod_manager = ProdManager(objs_pool.trainer, objs_pool.recommender, objs_pool.standard_saver, "model_example",
                                   assistant_builder=assist_builder, dataloader_builder=objs_pool.loader_builder,
                                   try_to_load=False)
        pass

    def test_update_with_new_users_get_recommends_for_them(self):
        # TODO when'll end up with recommender
        ...

    def test_full_pipeline(self):
        # TODO when'll end up with recommender
        ...

    def _create_standard_manager(self):
        assist_builder = AssistantBuilder(MFWithBiasModel, nusers=10, nitems=10, hidden_size=5)
        prod_manager = ProdManager(objs_pool.trainer, objs_pool.recommender, objs_pool.standard_saver, "model_example",
                                   assistant_builder=assist_builder, dataloader_builder=objs_pool.loader_builder)
        return prod_manager

    def _validate_recommends_format(self, recommends, nb_users):
        self.assertEqual(len(recommends), nb_users)
        self.assertGreater(len(recommends[0]), 0, "recommendations for user shouldn't be empty list!")
        self.assertIsInstance(recommends[0][0], int)