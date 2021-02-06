import unittest

from recsys_pipeline.main_tasks.prod_manager import ProdManager
from recsys_pipeline.model_level.assistant_builders import AssistantBuilder
from recsys_pipeline.model_level.models.mf_with_bias import MFWithBiasModel
from ..helpers import std_objects
from ..helpers.objs_pool import ObjsPool
from ..helpers.tests_config import TestsConfig
objs_pool = ObjsPool()
config = TestsConfig()


class TestTrainProdManager(unittest.TestCase):
    def test_add_interacts(self):
        prod_manager = self._create_standard_manager()
        interacts = std_objects.get_interacts()
        prod_manager.add_interacts(interacts)

    def test_fit(self):
        prod_manager = self._create_standard_manager()
        interacts = std_objects.get_interacts()
        prod_manager.add_interacts(interacts)
        prod_manager.fit(nepochs=2)


    def _create_standard_manager(self):
        assist_builder = AssistantBuilder(MFWithBiasModel, nusers=10, nitems=10, hidden_size=5)
        prod_manager = ProdManager(objs_pool.trainer, objs_pool.recommender, objs_pool.standard_saver, "model_example",
                                   assistant_builder=assist_builder)
        return prod_manager