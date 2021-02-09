import unittest
import torch
from torch import nn

from recsys_pipeline.model_level.data_processing.id_idx_conv import IdIdxConv
from ..helpers.objs_pool import ObjsPool
from ..helpers import std_objects, tests_config
objs_pool = ObjsPool()
config = tests_config.TestsConfig()


class TestModelAssistant(unittest.TestCase):
    def setUp(self):
        self.item_colname = config.item_colname

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
        self.assertLessEqual(len(all_items), len(interacts))

    def test_update_and_convert(self):
        assistant = std_objects.get_assistant(0, 0, 0)
        old_nb_users = len(assistant.get_all_items())
        old_init_kwargs = assistant.get_model_init_kwargs()
        self.assertEqual(old_nb_users, 0)

        interacts = std_objects.get_interacts(100)
        uniq_items = interacts[self.item_colname].unique()
        assistant.update_with_interacts(interacts)

        new_nb_users = len(assistant.get_all_items())
        new_init_kwargs = assistant.get_model_init_kwargs()
        self.assertEqual(new_nb_users, len(uniq_items))
        for init_kwarg_name in ["nusers", "nitems"]:
            old_val = old_init_kwargs[init_kwarg_name]
            new_val = new_init_kwargs[init_kwarg_name]
            self.assertGreater(new_val, old_val, "assistant doesn't update model init kwargs")

    def test_get_model(self):
        assistant = std_objects.get_assistant()
        model = assistant.get_model()
        self.assertIsInstance(model, nn.Module)

    def test_get_model_init_kwargs(self):
        assistant = std_objects.get_assistant()
        init_kwargs = assistant.get_model_init_kwargs()
        self.assertIsInstance(init_kwargs, dict)
        for key in init_kwargs.keys():
            self.assertIsInstance(key, str, "kwargs keys should be strings")

    def test_get_convs(self):
        assistant = std_objects.get_assistant()
        convs = assistant.get_convs()
        for cnv in convs:
            self.assertIsInstance(cnv, IdIdxConv)