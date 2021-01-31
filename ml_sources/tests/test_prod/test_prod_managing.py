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
from ..helpers import tests_config
config = tests_config.TestsConfig()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


class TestProdManager(unittest.TestCase):
    def setUp(self):
        self.default_models_dir = os.path.join(config.save_dir, "models/")
        self.saver_params_file_name = "prod_models_info.json"

    def test_manager_initialization(self):
        prod_manager = self._create_default_prod_manager()

    def test_add_interacts(self):
        prod_manager = self._create_default_prod_manager()
        new_interacts = self.load_ratings(50)
        prod_manager.add_interacts(new_interacts)

    def test_train(self):
        prod_manager = self._create_default_prod_manager()
        prod_manager.train_model(nepochs=2)

    def test_get_recommends(self):
        prod_manager = self._create_default_prod_manager()
        prod_manager.train_model(nepochs=2)
        users = [0, 1, 2]
        recommends = prod_manager.get_recommends(users)
        print("WOW recommends:", recommends)

    def test_save_results(self):
        raise NotImplementedError

    def test_update_trainer_with_interacts(self):
        raise NotImplementedError

    def _create_default_prod_manager(self):
        ratings = self.load_ratings()
        user_id_conv = id_idx_converter.IdIdxConverter()
        item_id_conv = id_idx_converter.IdIdxConverter()
        model_saver = meta_model_saving.MetaModelSaver(save_dir=self.default_models_dir,
                                                  params_file_name=self.saver_params_file_name,
                                                  )
        model_creator = mf_with_bias.MFWithBiasModel
        preprocessor = preprocessing.DataPreprocessor(DEVICE)
        dataloader_builder = loader_build.StandardLoaderBuilder(batch_size=16)

        prod_manager = prod_managing.ProdManager("testing_model_sample", 
                                                start_interacts=ratings, 
                                                user_id_idx_conv=user_id_conv,
                                                item_id_idx_conv=item_id_conv,
                                                model_saver=model_saver, 
                                                preprocessor=preprocessor,
                                                dataloader_builder=dataloader_builder, 
                                                train_kwargs={"lr": 2e-4},
                                                model_builder=model_creator,
                                                model_init_kwargs={"nusers": 50, "nitems": 50})
        return prod_manager

    def load_ratings(self, nrows=20):
        return pd.read_csv(config.interacts_path, nrows=nrows)