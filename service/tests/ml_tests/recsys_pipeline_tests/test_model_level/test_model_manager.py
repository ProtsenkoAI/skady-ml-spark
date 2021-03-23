import unittest
import torch
from torch import nn

from model_level.model_managing import ModelManager

from helpers import tests_config, std_objects

config = tests_config.TestsConfig()


class TestModelManager(unittest.TestCase):
    def setUp(self):
        self.item_colname = config.item_colname

    def test_preproc_then_forward(self):
        assistant =  std_objects.get_assistant()
        interacts = std_objects.get_interacts()
        assistant.update_with_interacts(interacts)

        dataloader =  std_objects.get_dataloader()
        batch = next(iter(dataloader))
        features, labels = batch
        preds, proc_labels = assistant.preproc_forward(features, labels=labels)
        self.assertEqual(len(preds), len(features))
        self.assertIsInstance(preds, torch.Tensor)
        self.assertEqual(len(proc_labels), len(preds))
        self.assertIsInstance(proc_labels, torch.Tensor)

    def test_update_with_interacts(self):
        assistant = std_objects.get_assistant(0, 0, 0)
        interacts = std_objects.get_interacts(100)
        assistant.update_with_interacts(interacts)

    def test_get_model(self):
        assistant = std_objects.get_assistant()
        model = assistant.get_model_manager()
        self.assertIsInstance(model, nn.Module)

    def test_update_then_predict(self):
        assistant = std_objects.get_assistant(nusers=1)
        old_interacts = std_objects.get_interacts(20)
        assistant.update_with_interacts(old_interacts)

        interacts = std_objects.get_interacts(200)
        assistant.update_with_interacts(interacts)

        dataloader = std_objects.get_dataloader(interacts=interacts)
        for features, labels in dataloader:
            assistant.preproc_forward(features)

    def test_save(self):
        saver = std_objects.get_standard_saver()
        assistant = std_objects.get_assistant(nusers=33)
        save_name = assistant.save(saver)
        self.assertIsInstance(save_name, str)

    def test_load_from_save(self):
        assistant = std_objects.get_assistant(nusers=3)
        saver = std_objects.get_standard_saver()
        save_name = assistant.save(saver)
        del assistant
        loaded_assist = ModelManager.from_save(saver, save_name)
        self.assertIsInstance(loaded_assist, ModelManager)
