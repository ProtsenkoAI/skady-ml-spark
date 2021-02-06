import unittest

from recsys_pipeline.main_tasks.prod_manager import ProdManager
from ..helpers.objs_pool import ObjsPool
objs_pool = ObjsPool()


class TestTrainProdManager(unittest.TestCase):
    def test_add_interacts(self):
        prod_manager = ProdManager(objs_pool.trainer, objs_pool.recommender, objs_pool.standard_saver, "model_example")
