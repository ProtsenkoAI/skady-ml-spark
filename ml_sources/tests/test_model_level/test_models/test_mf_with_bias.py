import unittest
import torch

from recsys_pipeline.model_level.models.mf_with_bias import MFWithBiasModel
from ...helpers.objs_pool import ObjsPool
from ...helpers import std_objects
objs_pool = ObjsPool()


class TestMFWithBias(unittest.TestCase):
    def test_forward(self):
        model = std_objects.get_model(5, 5, 5)
        batch_size = 4

        interacts = std_objects.get_interacts()
        loader = std_objects.get_dataloader(batch_size, interacts)
        preprocessor = std_objects.get_processor()
        preprocessor.update(interacts)

        features, labels = next(iter(loader))
        batch_proc = preprocessor.preprocess_features(features)

        preds = model.forward(*batch_proc)
        self.assertIsInstance(preds, torch.Tensor)
        self.assertEqual(len(preds), len(features))

    def test_add_users(self):
        src_users = 4
        added_users = 13

        model = std_objects.get_model(src_users, 3, 10)
        model.add_users(added_users)

        nusers_in_model = model.get_init_kwargs()["nusers"]
        self.assertEqual(src_users + added_users, nusers_in_model)

    def test_add_items(self):
        src_items = 5
        added_items = 16
        model = std_objects.get_model(5, src_items, 10)
        model.add_items(added_items)
        nusers_in_model = model.get_init_kwargs()["nitems"]
        self.assertEqual(src_items + added_items, nusers_in_model)

    def test_get_init_kwargs(self):
        src_init_kwargs = {"nusers": 10, "nitems": 11, "hidden_size": 13}
        model = MFWithBiasModel(**src_init_kwargs)
        loaded_kwargs = model.get_init_kwargs()
        for key in src_init_kwargs.keys():
            self.assertEqual(src_init_kwargs[key], loaded_kwargs[key])
