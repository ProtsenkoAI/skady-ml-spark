import os

from ..data.building_loaders import StandardLoaderBuilder, UserItemsLoaderBuilder
from ..train_eval.training import SparkTrainer, SimpleTrainer
from ..model_level.data_processing import DataProcessor, SparkProcessor, IdIdxConv, TensorCreator
from ..model_level.models import MFWithBiasModel
from ..model_level.model_managing import ModelManager
from ..model_level.recommender import Recommender
from ..data.obtain import SparkObtainer, SimpleObtainer
from ..saving.savers import SparkSaver


class ObjsCreator:
    def __init__(self, config):
        self.config = config
        self.mode = config["mode"]
        self.device = config["device"]
        self.model_path = os.path.join(config["base_path"], config["worker_dir"], config["model_checkpoint_name"])
        self.spark_saver = SparkSaver(self.model_path)

        train_params = config["train_params"]
        self.batch_size = train_params["batch_size"]

        if self.mode not in ["spark", "local"]:
            raise ValueError(self.mode)

    def get_fitter(self):
        # TODO: add typings and Trainer interface
        loader_builder = StandardLoaderBuilder(self.batch_size)
        if self.mode == "spark":
            # TODO: replace torch' DataLoader with custom data iterator implementation (maybe?)
            fitter = SparkTrainer(loader_builder, self.spark_saver)
        elif self.mode == "local":
            fitter = SimpleTrainer(loader_builder)
        return fitter

    def get_model_manager(self):
        # TODO: add ability to load manager
        # TODO: add force_create flag for model
        user_conv = IdIdxConv()
        item_conv = IdIdxConv()
        tensor_creator = TensorCreator(device=self.device)
        if self.mode == "local":
            processor = DataProcessor(user_conv, item_conv, tensor_creator)
        elif self.mode == "spark":
            processor = SparkProcessor(user_conv, item_conv, tensor_creator)

        # TODO: instead of hardcoded model, can pass model_params to model fabric.
        torch_model = MFWithBiasModel(**self.config["model_params"])
        # torch_model = self.spark_saver.load_or_create(self.config["model_checkpoint_name"])
        model_manager = ModelManager(torch_model, processor)
        return model_manager

    def get_data_obtainer(self):
        if self.mode == "local":
            return SimpleObtainer(self.config)

        elif self.mode == "spark":
            return SparkObtainer(self.config)

    def get_recommender(self):
        load_builder = UserItemsLoaderBuilder(batch_size=self.config["recommend_params"]["batch_size"])
        return Recommender(load_builder)
