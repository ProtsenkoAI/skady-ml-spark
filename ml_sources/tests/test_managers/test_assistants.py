import unittest
import torch

from ..helpers import objects_creation
from recsys_pipeline.managers import assistants


class TestModelAssistant(unittest.TestCase):
    def setUp(self):
        self.user_colname = "user_id"
        self.label_colname = "rating"
        self.std_nrows = 100
        self.std_batch = 8
        self.std_interacts = objects_creation.get_interacts(nrows=self.std_nrows)
        self.std_model = objects_creation.get_mf_model()
        self.std_assistant = assistants.ModelAssistant()
        self.std_assistant.update_with_new_interacts(self.std_interacts)

        self.std_dataset = objects_creation.get_interact_dataset(nrows=self.std_nrows)
        self.std_dataloader = objects_creation.create_loader_from_dataset(self.std_dataset, batch_size=self.std_batch)

    def test_forward(self):
        features, labels = next(iter(self.std_dataloader))

        preds = self.std_assistant.preproc_then_forward(self.std_model, features)
        self.assertEqual(len(preds), self.std_batch)
        self.assertIsInstance(preds, torch.Tensor)

    def test_preproc_labels(self):
        labels = self.std_interacts[self.label_colname]
        proc_labels = self.std_assistant.preproc_labels(labels)
        self.assertEqual(len(labels), len(proc_labels))
        self.assertIsInstance(proc_labels, torch.Tensor)
