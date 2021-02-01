import unittest

from ..helpers import objects_creation
from recsys_pipeline.managers import trainers


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.nusers = 50
        self.nitems = 50
        self.interacts_nrows = 20
        self.preprocessor = objects_creation.get_preprocessor()

    def test_train_2_epochs(self):
        trainer = self._create_trainer()
        training_epochs = 2
        trainer.fit(nepochs=training_epochs)

    def test_train_5_steps(self):
        trainer = self._create_trainer()
        training_steps = 5
        trainer.fit(nsteps=training_steps)

    def test_every_layer_is_updated(self):
        trainer = self._create_trainer()
        old_weights = self._get_model_weights(trainer)
        trainer.fit(nsteps=1)
        new_weights = self._get_model_weights(trainer)

        for old, new in zip(old_weights, new_weights):
            old_and_new_are_identical = (old == new).all()
            self.assertFalse(old_and_new_are_identical)
    
    def test_add_user(self):
        interacts = objects_creation.get_interacts(self.interacts_nrows)
        interacts["user_id"][0] = self.nusers
        # new_dataloader = helpers.create_dataloader(self.interacts)
        trainer = self._create_trainer_with_interacts(interacts)
        trainer.add_users(1)
        # and trying to train model with it
        trainer.fit(nepochs=2)

    def test_add_multiple_users(self):
        interacts = objects_creation.get_interacts(self.interacts_nrows)
        users_added = 6

        for idx, new_user_idx in enumerate(range(self.nusers, self.nusers + users_added)):
            interacts["user_id"][idx] = new_user_idx

        trainer = self._create_trainer_with_interacts(interacts)
        trainer.add_users(users_added)
        trainer.fit(nepochs=2)

    def test_add_item(self):
        interacts = objects_creation.get_interacts(self.interacts_nrows)
        interacts["anime_id"][0] = self.nitems
        # new_dataloader = helpers.create_dataloader(self.interacts)
        trainer = self._create_trainer_with_interacts(interacts)
        trainer.add_items(1)
        # and trying to train model with it
        trainer.fit(nepochs=2)

    def test_add_multiple_items(self):
        interacts = objects_creation.get_interacts(self.interacts_nrows)
        items_added = 6
        for idx, new_item_idx in enumerate(range(self.nitems, self.nitems + items_added)):
            interacts["anime_id"][idx] = new_item_idx

        trainer = self._create_trainer_with_interacts(interacts)
        trainer.add_items(items_added)
        trainer.fit(nepochs=2)

    def test_get_recommends_for_users(self):
        interacts = objects_creation.get_interacts()
        trainer = self._create_trainer_with_interacts(interacts)
        users = interacts["user_id"][:5]
        items = interacts["anime_id"].unique()

        proc_users = self.preprocessor.preprocess_users(users)
        proc_items = self.preprocessor.preprocess_items(items)
        print(proc_users.shape, proc_items.shape)

        recommends = trainer.get_recommends_for_users(proc_users, proc_items)
        self.assertEqual(len(recommends), len(proc_users))
        self.assertGreater(len(recommends[0]), 0)

    def test_get_model(self):
        trainer = self._create_trainer()
        model = trainer.get_model()
    
    def _get_model_weights(self, trainer):
        weights = []
        for param in trainer.get_model().parameters():
            param_weights = param.detach().numpy().copy()
            weights.append(param_weights)

        return weights

    def _create_trainer(self):
        interacts = objects_creation.get_interacts(self.interacts_nrows)
        trainer = self._create_trainer_with_interacts(interacts)
        return trainer

    def _create_trainer_with_interacts(self, interacts):
        model = objects_creation.get_mf_model(self.nusers, self.nitems)
        dataset = objects_creation.create_dataset_from_interacts(interacts)
        dataloader = objects_creation.create_loader_from_dataset(dataset)
        trainer = trainers.Trainer(model, dataloader, self.preprocessor)
        return trainer
