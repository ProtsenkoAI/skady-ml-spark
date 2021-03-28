# TODO: check how the sparktorch repo is implemented on the github for insights

from train_eval.trainers.streaming_trainer import StreamingTrainer
from .fit_objs.spark_fit_obj import SparkFitObj
from model_level import ModelManager
import os


class SparkStreamingTrainer(StreamingTrainer):
    def __init__(self, params, save_dir):
        self.params = params
        self.save_dir = save_dir
        super().__init__()

    def get_fit_obj(self, manager: ModelManager) -> SparkFitObj:
        return SparkFitObj(manager, self.params, self.save_dir)
