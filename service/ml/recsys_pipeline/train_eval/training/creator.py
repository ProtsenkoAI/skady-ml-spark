import os

from saving.savers import SparkSaver
from .spark_trainer import StreamingTrainer
from .simple_trainer import SimpleTrainer
from data.building_loaders import StandardLoaderBuilder


class TrainerCreator:
    def __init__(self, params, paths_info, mode):
        self.mode = mode

        self.model_path = os.path.join(paths_info["base_path"], paths_info["worker_dir"], params["model_checkpoint_name"])
        self.spark_saver = SparkSaver(self.model_path)

        self.batch_size = params["batch_size"]

    def get(self):
        # TODO: add typings and Trainer interface
        loader_builder = StandardLoaderBuilder(self.batch_size)
        if self.mode == "spark":
            # TODO: replace torch' DataLoader with custom data iterator implementation (maybe?)
            fitter = StreamingTrainer(loader_builder, self.spark_saver)
        elif self.mode == "local":
            fitter = SimpleTrainer(loader_builder)
        else:
            raise ValueError(f"mode is incorrect: {self.mode}")
        return fitter
