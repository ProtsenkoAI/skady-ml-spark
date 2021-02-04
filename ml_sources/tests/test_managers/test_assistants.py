import unittest
import torch

from ..helpers import objects_creation
from recsys_pipeline.managers import assistants


class TestModelAssistant(unittest.TestCase):
    def setUp(self):
        self.user_colname = "user_id"

    def test_forward(self):
        batch_size = 10
        interacts = objects_creation.get_interacts(nrows=10)
        dataset = objects_creation.get_interact_dataset(nrows=10)
        dataloader = objects_creation.create_loader_from_dataset(dataset, batch_size=batch_size)
        features, labels = next(iter(dataloader))

        model = objects_creation.get_mf_model()
        assistant = assistants.ModelAssistant(model)
        assistant.update_with_new_interacts(interacts)
        preds = assistant.preproc_then_forward(features)
        self.assertEqual(len(preds), batch_size)
        self.assertIsInstance(preds, torch.Tensor)

    def test_get_recommends_assist(self):
        interacts = objects_creation.get_interacts(nrows=100)
        model = objects_creation.get_mf_model()
        assistant = assistants.ModelAssistant(model)
        assistant.update_with_new_interacts(interacts)

        some_users = interacts[self.user_colname].unique()[:10]
        recommends = assistant.get_recommends(some_users)
        self.assertEqual(len(recommends), len(some_users))
        self.assertEqual(model.get_init_kwargs()["nitems"], len(recommends[0])) # test that every item is in recommends

    def test_get_probas_for_every_item(self):
        raise NotImplementedError

    def test_preproc_labels(self):
        raise NotImplementedError