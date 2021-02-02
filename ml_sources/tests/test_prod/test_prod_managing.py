import unittest
import torch
import pandas as pd
import os

import prod_managing
from recsys_pipeline.models import mf_with_bias
from recsys_pipeline.saving import model_with_meta_and_ids_saving
from recsys_pipeline.data_transform import preprocessing
from recsys_pipeline.data import loader_factories
from ..helpers import tests_config, objects_creation

config = tests_config.TestsConfig()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestProdManager(unittest.TestCase):
    def setUp(self):
        self.nrows = 20
        self.nusers = 50
        self.nitems = 50
        self.default_models_dir = os.path.join(config.save_dir, "models/")
        self.saver_params_file_name = "prod_models_info.json"
        self.model_name = "testing_model_sample"
        self.model_creator = mf_with_bias.MFWithBiasModel

        self.user_colname = "user_id"

    def test_manager_initialization(self):
        prod_manager = self._create_default_prod_manager()

    def test_add_interacts(self):
        prod_manager = self._create_default_prod_manager()
        new_interacts = objects_creation.get_interacts(50)
        prod_manager.add_interacts(new_interacts)

    def test_train(self):
        prod_manager = self._create_default_prod_manager()
        prod_manager.fit(nepochs=2)

    def test_get_recommends(self):
        prod_manager = self._create_default_prod_manager()
        prod_manager.fit(nepochs=2)
        users = objects_creation.get_interacts(20)[self.user_colname].unique()[:3] # get 3 users from interacts
        recommends = prod_manager.get_recommends(users)
        self._validate_recommends_format(recommends, len(users))

    def test_save_results(self):
        prod_manager = self._create_default_prod_manager()
        prod_manager.save()
        saver = self._create_default_saver()
        model_saved = saver.check_model_exists()
        self.assertTrue(model_saved, "Prod Manager didn't save model after calling save_results()")

    def test_save_then_load(self):
        prod_manager = self._create_default_prod_manager()
        prod_manager.save()
        self.model_creator = None # so manager can't create new model: only load saved one
        new_manager = self._create_default_prod_manager()
        # saving again to check that the model functions properly
        new_manager.save()

    def test_create_new_objects_if_cant_load(self):
        class FakeModelBuilder:
            def __call__(self, *args, **kwargs):
                raise ValueError("Calling FakeModelBuilder, so we know: manager tries to create new one!")

        self.model_creator = FakeModelBuilder()
        with self.assertRaises(ValueError):
            prod_manager = self._create_default_prod_manager()

    def test_update_trainer_with_interacts(self):
        new_interacts = objects_creation.get_interacts(self.nrows + 100)
        prod_manager = self._create_default_prod_manager()
        prod_manager.add_interacts(new_interacts)
        new_user = new_interacts[self.user_colname].max()
        new_users = [new_user]
        prod_manager.fit(nsteps=2)
        new_users_recommends = prod_manager.get_recommends(new_users)
        self._validate_recommends_format(new_users_recommends, len(new_users))

    def test_full_pipeline(self):
        """
        1. Create ProdManager from scratch
        2. Add interacts
        3. Train model
        4. add some more interacts
        5. Train again
        6. Save
        7. Create new ProdManager using save from previous step
        8. Make a recommendation
        """
        manager = self._create_default_prod_manager(try_to_load=False)
        new_interacts = objects_creation.get_interacts(self.nrows + 10)
        manager.add_interacts(new_interacts)
        manager.fit(nsteps=5)

        second_new_interacts = objects_creation.get_interacts(self.nrows + 15)
        manager.add_interacts(second_new_interacts)
        manager.fit(nsteps=5)
        manager.save()

        loaded_manager = self._create_default_prod_manager(try_to_load=True)
        some_users = new_interacts[self.user_colname].unique()[:3]
        recommends = loaded_manager.get_recommends(some_users)

        self._validate_recommends_format(recommends, len(some_users))

    def _create_default_prod_manager(self, try_to_load=True):
        interacts = objects_creation.get_interacts(self.nrows)
        model_saver = self._create_default_saver()
        preprocessor = preprocessing.TensorCreator(DEVICE)
        dataloader_builder = loader_factories.StandardLoaderBuilder(batch_size=16)

        prod_manager = prod_managing.ProdManager(model_saver=model_saver,
                                                 preprocessor=preprocessor,
                                                 dataloader_builder=dataloader_builder,
                                                 train_kwargs={"lr": 2e-4},
                                                 model_builder=self.model_creator,
                                                 model_init_kwargs={"nusers": self.nusers, "nitems": self.nitems},
                                                 try_to_load_model=try_to_load)
        prod_manager.add_interacts(interacts)
        return prod_manager

    def _create_default_saver(self):
        return model_with_meta_and_ids_saving.ModelAndIdsSaver(model_name=self.model_name,
                                                               save_dir=self.default_models_dir,
                                                               params_file_name=self.saver_params_file_name,
                                                               )

    def load_ratings(self, nrows=20):
        return pd.read_csv(config.interacts_path, nrows=nrows)

    def _validate_recommends_format(self, recommends, nb_users):
        self.assertEqual(len(recommends), nb_users)
        self.assertGreater(len(recommends[0]), 0, "recommendations for user shouldn't be empty list!")
        self.assertIsInstance(recommends[0][0], int)