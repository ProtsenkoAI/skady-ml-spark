import unittest
import torch
from torch import nn

from recsys_pipeline.model_level.data_processing.id_idx_conv import IdIdxConv
from ...helpers.objs_pool import ObjsPool
from ...helpers import std_objects, tests_config
objs_pool = ObjsPool()
config = tests_config.TestsConfig()


class TestDataProcessing(unittest.TestCase):
    def test_update(self):
        batch_size = 6

        processor = std_objects.get_processor()
        interacts = std_objects.get_interacts()
        processor.update(interacts)

    def test_preproc_features(self):
        batch_size = 6

        processor = std_objects.get_processor()
        interacts = std_objects.get_interacts()
        processor.update(interacts)
        # check that now we can preproc interacts
        loader = std_objects.get_dataloader(batch_size=batch_size, interacts=interacts)
        features, _ = next(iter(loader))
        users, items = processor.preprocess_features(features)

        for feature_type in [users, items]:
            self.assertIsInstance(feature_type, torch.Tensor)
            self.assertEqual(len(feature_type), batch_size)

    def test_preproc_labels(self):
        batch_size = 6

        processor = std_objects.get_processor()
        interacts = std_objects.get_interacts()
        processor.update(interacts)
        # check that now we can preproc interacts
        loader = std_objects.get_dataloader(batch_size=batch_size, interacts=interacts)
        _, labels = next(iter(loader))
        labels_proc = processor.preprocess_labels(labels)

        self.assertIsInstance(labels_proc, torch.Tensor)
        self.assertEqual(len(labels_proc), batch_size)

    def test_get_convs(self):
        processor = std_objects.get_processor()
        interacts = std_objects.get_interacts()
        processor.update(interacts)

        user_conv = processor.get_user_conv()
        item_conv = processor.get_item_conv()
        self.assertIsInstance(user_conv, IdIdxConv)
        self.assertIsInstance(item_conv, IdIdxConv)

    def test_count_unknowns(self):
        processor = std_objects.get_processor()
        interacts = std_objects.get_interacts(20)
        nusers, nitems = processor.count_unknown_users_items(interacts)
        self.assertGreater(nusers, 0)
        self.assertGreater(nitems, 0)

        lot_of_interacts = std_objects.get_interacts(100)
        new_nusers, new_nitems = processor.count_unknown_users_items(lot_of_interacts)
        self.assertGreaterEqual(new_nusers, nusers)
        self.assertGreaterEqual(new_nitems, nitems)