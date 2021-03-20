import os

from ..data.building_loaders import StandardLoaderBuilder
from ..train_eval.training import SparkTrainer, SimpleTrainer
from ..model_level.data_processing import DataProcessor, SparkProcessor, IdIdxConv, TensorCreator
from ..model_level.models import MFWithBiasModel
from ..model_level.model_managing import ModelManager
from ..data.obtain import SparkObtainer, SimpleObtainer
from ..saving.savers import SparkSaver


class ObjsCreator:
    def __init__(self, config):
        self.config = config
        self.mode = config["mode"]
        self.device = config["device"]
        self.model_path = os.path.join(config["base_path"], config["worker_dir"], config["model_file_name"])

        train_params = config["train_params"]
        self.batch_size = train_params["batch_size"]

        if self.mode not in ["spark", "local"]:
            raise ValueError(self.mode)

    def get_fitter(self):
        # TODO: add typings and Trainer interface
        loader_builder = StandardLoaderBuilder(self.batch_size)
        spark_saver = SparkSaver(self.model_path)
        if self.mode == "spark":
            fitter = SparkTrainer(loader_builder, spark_saver)
        elif self.mode == "local":
            fitter = SimpleTrainer(loader_builder)
        return fitter

    def get_model(self):
        user_conv = IdIdxConv()
        item_conv = IdIdxConv()
        tensor_creator = TensorCreator(device=self.device)
        if self.mode == "local":
            processor = DataProcessor(user_conv, item_conv, tensor_creator)
        elif self.mode == "spark":
            processor = SparkProcessor(user_conv, item_conv, tensor_creator)

        torch_model = MFWithBiasModel.load_or_create(self.model_path)
        model_manager = ModelManager(torch_model, processor)
        return model_manager

    def get_data_obtainer(self):
        if self.mode == "local":
            return SimpleObtainer(self.config)

        elif self.mode == "spark":
            return SparkObtainer(self.config)
