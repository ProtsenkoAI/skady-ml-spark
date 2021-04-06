import os
from model.expose.saver import Saver
from model.saving import SparkSaver
from model.saving.spark_manager_saver import SparkManagerSaver


class SaverCreator:
    def __init__(self, config, common_params):
        mode = common_params["mode"]
        if mode not in ["spark"]:
            raise ValueError(mode)
        self.paths = common_params["paths"]

    def get_train_obj_saver(self) -> Saver:
        model_path = os.path.join(self.paths["base_path"],
                                  self.paths["worker_dir"],
                                  self.paths["packaged_train_obj_name"])
        return SparkSaver(model_path)

    def get_manager_saver(self) -> SparkManagerSaver:
        model_path = os.path.join(self.paths["base_path"],
                                  self.paths["worker_dir"],
                                  self.paths["model_checkpoint_name"])
        return SparkManagerSaver(model_path)
