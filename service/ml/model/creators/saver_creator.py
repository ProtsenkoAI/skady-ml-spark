import os
from model.expose.saver import Saver
from model.saving.spark_saver import SparkSaver


class SaverCreator:
    def __init__(self, config, common_params):
        mode = common_params["mode"]
        if mode not in ["spark"]:
            raise ValueError(mode)
        self.paths = common_params["paths"]

    def get(self) -> Saver:
        model_path = os.path.join(self.paths["base_path"],
                                  self.paths["worker_dir"],
                                  self.paths["model_checkpoint_name"])
        return SparkSaver(model_path)
