from .fit_obj import FitObj
from saving.savers import SparkSaver
from model_level import ModelManager
from data import StandardLoaderBuilder
from train_eval.updating_weights.weights_updater import WeightsUpdater
import os
import pandas as pd


class SparkFitObj(FitObj):
    def __init__(self, manager: ModelManager, params: dict, save_dir):
        save_path = os.path.join(save_dir, params["model_checkpoint_name"])

        self.spark_saver = SparkSaver(save_path)
        weights_updater = WeightsUpdater(params["lr"])
        loader_builder = StandardLoaderBuilder(params["batch_size"])
        self.spark_saver.save(manager, weights_updater, loader_builder)

    def __call__(self, batch: pd.DataFrame, **kwargs):
        manager, weights_updater, loader_builder = self.spark_saver.load()

        weights_updater.prepare_for_fit(manager)
        batch_iterator = loader_builder.build(batch)
        for minibatch in batch_iterator:
            weights_updater.fit_with_batch(manager, minibatch)

        self.spark_saver.save(manager, weights_updater, loader_builder)
