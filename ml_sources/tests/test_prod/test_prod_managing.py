# TODO
import unittest
import torch
import pandas as pd
import os

from prod import prod_managing
from recsys_pipeline.models import mf_with_bias
from recsys_pipeline.saving import meta_model_saving
from recsys_pipeline.data_transform import id_idx_converter, preprocessing
from recsys_pipeline.data import loader_build
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

    def test_manager_initialization(self):
        prod_manager = self._create_default_prod_manager()

    def test_add_interacts(self):
        prod_manager = self._create_default_prod_manager()
        new_interacts = objects_creation.get_interacts(50)
        prod_manager.add_interacts(new_interacts)

    def test_train(self):
        prod_manager = self._create_default_prod_manager()
        prod_manager.train_model(nepochs=2)

    def test_get_recommends(self):
        prod_manager = self._create_default_prod_manager()
        prod_manager.train_model(nepochs=2)
        users = [0, 1, 2]
        recommends = prod_manager.get_recommends(users)
        self.assertEqual(len(recommends), len(users))
        self.assertGreater(len(recommends[0]), 0, "recommendations for user shouldn't be empty list!")
        self.assertIsInstance(recommends[0][0], int)

    def test_save_results(self):
        prod_manager = self._create_default_prod_manager()
        prod_manager.save_results()
        saver = self._create_default_saver()
        model_saved = saver.check_model_exists(self.model_name)
        self.assertTrue(model_saved, "Prod Manager didn't save model after calling save_results()")

    def test_save_then_load(self):
        prod_manager = self._create_default_prod_manager()
        prod_manager.save_results()
        self.model_creator = None # so manager can't create new model: only load saved one
        new_manager = self._create_default_prod_manager()
        # saving again to check that the model functions properly
        new_manager.save_results()

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
        prod_manager.update_trainer_with_interacts(new_interacts)
        new_user = self.nusers
        new_users = [new_user]
        new_users_recommends = prod_manager.get_recommends(new_users)
        print("WOW! new_users_recommends", new_users_recommends)
        self.assertEqual(len(new_users_recommends), len(new_users))
        self.assertGreater(len(new_users_recommends[0]), 0, "recommendations for user shouldn't be empty list!")
        self.assertIsInstance(new_users_recommends[0][0], int)

    def test_full_pipeline(self):
        raise NotImplementedError

    def _create_default_prod_manager(self):
        interacts = objects_creation.get_interacts(self.nrows)
        model_saver = self._create_default_saver()
        preprocessor = preprocessing.DataPreprocessor(DEVICE)
        dataloader_builder = loader_build.StandardLoaderBuilder(batch_size=16)

        prod_manager = prod_managing.ProdManager(self.model_name,
                                                 model_saver=model_saver,
                                                 preprocessor=preprocessor,
                                                 dataloader_builder=dataloader_builder,
                                                 train_kwargs={"lr": 2e-4},
                                                 model_builder=self.model_creator,
                                                 model_init_kwargs={"nusers": self.nusers, "nitems": self.nitems})
        prod_manager.add_interacts(interacts)
        return prod_manager

    def _create_default_saver(self):
        return meta_model_saving.MetaModelSaver(save_dir=self.default_models_dir,
                                                params_file_name=self.saver_params_file_name,
                                                )

    def load_ratings(self, nrows=20):
        return pd.read_csv(config.interacts_path, nrows=nrows)
