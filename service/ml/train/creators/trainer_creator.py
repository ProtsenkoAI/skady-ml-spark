import os
from ..trainers.streaming_trainer_impl import StreamingTrainerImpl, StreamingTrainer

from model.creators.saver_creator import SaverCreator
from model import UsersManagerCreator
from train.trainers.fit_objs import SparkFitObj
from ..updating_weights.weights_updater_impl import WeightsUpdaterImpl
from data.building_loaders import StandardLoaderBuilder
from model.creators import ManagerCreator
from model.expose import ModelManager


class TrainerCreator:
    # TODO: can delete spark_streaming_trainer and union in with streaming_trainer
    def __init__(self, config, common_params):
        self.config = config
        self.common_params = common_params
        self.params = config["fitter_params"]

        self.users_manager_params = config["users_manager_params"]
        paths_info = common_params["paths"]
        self.trainer_params = {
            "lr": self.params["lr"],
            "batch_size": self.params["batch_size"],
            "nepochs": self.params["nepochs"]
        }
        self.mode = common_params["mode"]
        self.model_file = paths_info["model_checkpoint_name"]

        if self.mode == "spark":
            self.worker_dir = os.path.join(paths_info["base_path"],
                                           paths_info["worker_dir"])
        else:
            raise ValueError(self.mode)

    def get(self):
        raise NotImplementedError

    def get_streaming(self) -> StreamingTrainer:
        if self.mode == "spark":
            saver_creator = SaverCreator(self.config, self.common_params)
            saver = saver_creator.get_train_obj_saver()
            manager_saver = saver_creator.get_manager_saver()
            users_manager = UsersManagerCreator(self.config, self.common_params).get()
            fit_obj = SparkFitObj(users_manager, saver, ModelManager, manager_saver)

        weights_updater = WeightsUpdaterImpl(self.trainer_params["lr"])
        loader_builder = StandardLoaderBuilder(self.trainer_params["batch_size"])
        model_manager_creator = ManagerCreator(self.config, self.common_params)

        fitter = StreamingTrainerImpl(fit_obj, saver, weights_updater, loader_builder, model_manager_creator)
        return fitter
