import unittest
import pandas as pd
import torch
from torch.utils import data as torch_data

from . import helpers
from recsys_pipeline.managers import trainers


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.nusers = 50
        self.nitems = 50
        self.interacts_nrows = 20

    def test_train_2_epochs(self):
        interacts = helpers.load_interacts(self.interacts_nrows)
        trainer = self._create_trainer(interacts)
        training_epochs = 2
        training_steps = training_epochs * len(self.dataloader)
        trainer.fit(nepochs=training_epochs)
        epochs_passed = trainer.get_epochs_passed()
        steps_passed = trainer.get_steps_passed()

        self.assertEqual(training_epochs, epochs_passed)
        self.assertEqual(training_steps, steps_passed)

    def test_train_5_steps(self):
        interacts = helpers.load_interacts(10)
        trainer = self._create_trainer(interacts)
        training_steps = 5
        trainer.fit(nsteps=training_steps)
        epochs_passed = trainer.get_epochs_passed()
        steps_passed = trainer.get_steps_passed()

        self.assertEqual(epochs_passed, 2)
        self.assertEqual(steps_passed, 5)

    def test_every_layer_is_updated(self):
        interacts = helpers.load_interacts(self.interacts_nrows)
        trainer = self._create_trainer(interacts)
        old_weights = self._get_model_weights()
        trainer.fit(nsteps=1)
        new_weights = self._get_model_weights()

        for old, new in zip(old_weights, new_weights):
            old_and_new_are_identical = (old == new).all()
            self.assertFalse(old_and_new_are_identical)
    
    def test_add_user(self):
        interacts = helpers.load_interacts(self.interacts_nrows)
        interacts["user_id"][0] = self.nusers
        # new_dataloader = helpers.create_dataloader(self.interacts)
        trainer = self._create_trainer(interacts)
        trainer.add_users(1)
        # and trying to train model with it
        trainer.fit(nepochs=2)

    def test_add_multiple_users(self):
        interacts = helpers.load_interacts(self.interacts_nrows)
        users_added = 6

        for idx, new_user_idx in enumerate(range(self.nusers, self.nusers + users_added)):
            interacts["user_id"][idx] = new_user_idx

        trainer = self._create_trainer(interacts)
        trainer.add_users(users_added)
        trainer.fit(nepochs=2)

    def test_add_item(self):
        interacts = helpers.load_interacts(self.interacts_nrows)
        interacts["anime_id"][0] = self.nitems
        # new_dataloader = helpers.create_dataloader(self.interacts)
        trainer = self._create_trainer(interacts)
        trainer.add_items(1)
        # and trying to train model with it
        trainer.fit(nepochs=2)

    def test_add_multiple_items(self):
        interacts = helpers.load_interacts(self.interacts_nrows)
        items_added = 6

        for idx, new_item_idx in enumerate(range(self.nitems, self.nitems + items_added)):
            interacts["anime_id"][idx] = new_item_idx

        trainer = self._create_trainer(interacts)
        trainer.add_items(items_added)
        trainer.fit(nepochs=2)

    def test_get_recommends_for_users(self):
        interacts = helpers.load_interacts(self.interacts_nrows)
        trainer = self._create_trainer(interacts)
        users = interacts["user_id"][:5]
        items = interacts["anime_id"].unique()

        proc_users = self.preprocessor.preprocess_users(users)
        proc_items = self.preprocessor.preprocess_items(items)
        print("PROC one-to-many", proc_users, proc_items)
        print(proc_users.shape, proc_items.shape)

        trainer.get_recommends_for_users(proc_users, proc_items)

    def test_get_model(self):
        interacts = helpers.load_interacts(self.interacts_nrows)
        trainer = self._create_trainer(interacts)

        model = trainer.get_model()
    
    def _get_model_weights(self):
        weights = []
        for param in self.model.parameters():
            param_weights = param.detach().numpy().copy()
            weights.append(param_weights)

        return weights

    def _create_trainer(self, interacts):
        self.model = helpers.get_model(self.nusers, self.nitems)
        self.dataloader = helpers.create_dataloader(interacts)
        self.preprocessor = helpers.get_preprocessor()
        trainer = trainers.Trainer(self.model, self.dataloader, self.preprocessor)
        return trainer
