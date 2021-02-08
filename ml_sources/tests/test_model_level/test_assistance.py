import unittest
import torch

from ..helpers.objs_pool import ObjsPool
from ..helpers import std_objects, tests_config
objs_pool = ObjsPool()
config = tests_config.TestsConfig()


class TestModelAssistant(unittest.TestCase):
    def test_preproc_then_forward(self):
        assistant = objs_pool.assistant
        interacts = std_objects.get_interacts()
        assistant.update_with_interacts(interacts)

        dataloader = objs_pool.train_dataloader
        batch = next(iter(dataloader))
        features, labels = batch

        preds = assistant.preproc_then_forward(features)
        self.assertEqual(len(preds), len(features))
        self.assertIsInstance(preds, torch.Tensor)

    def test_preproc_labels(self):
        assistant = objs_pool.assistant
        interacts = std_objects.get_interacts()
        assistant.update_with_interacts(interacts)

        labels = interacts[config.labels_colname]
        labels_proc = assistant.preproc_labels(labels)

        self.assertEqual(len(labels_proc), len(labels))
        self.assertIsInstance(labels_proc, torch.Tensor)

    def test_get_all_items(self):
        assistant = objs_pool.assistant
        interacts = std_objects.get_interacts()
        assistant.update_with_interacts(interacts)
        all_items = assistant.get_all_items()


    def test_update_and_convert(self):
        ...

    def test_get_model(self):
        ...

    def test_get_model_init_kwargs(self):
        ...

    def test_get_convs(self):
        ...