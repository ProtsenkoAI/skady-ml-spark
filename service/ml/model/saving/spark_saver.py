# TODO: check for better ways for serializing strategies (refactor)
# TODO: check that serialises well
# TODO: test time to load and save model
import os
from . import util
from train.updating_weights.weights_updater_impl import WeightsUpdaterImpl
from .. import Saver
from ..expose.types import PackagedModel
from collections import namedtuple

SerializedPackagedModel = namedtuple("SerializablePackagedModel",
                                     ["optimizer_class", "criterion_class", "optim_params",
                                      "loader_creator"])


class SparkSaver(Saver):
    # TODO: refactor and test
    # TODO: not to hardcode classes
    def __init__(self, save_path):
        self.save_path = save_path

    def load_packaged_model(self) -> PackagedModel:
        if not self.check_saved():
            self._raise_not_saved()
        model_obj = self._load_fit_obj_from_path(self.save_path)

        weights_updater = WeightsUpdaterImpl(criterion_class=model_obj.criterion_class,
                                             optimizer_class=model_obj.optimizer_class,
                                             **model_obj.optim_params)
        loader_builder = model_obj.loader_creator
        return PackagedModel(weights_updater, loader_builder)

    def save_packaged_model(self, packaged_model: PackagedModel):
        updater, loader_creator = packaged_model.updater, packaged_model.loader_builder
        optimizer = updater.get_optimizer_class()
        criterion = updater.get_criterion_class()
        optim_params = updater.get_optim_params()
        obj_to_save = SerializedPackagedModel(
            criterion_class=criterion,
            optimizer_class=optimizer,
            optim_params=optim_params,
            loader_creator=loader_creator
        )

        util.dump_dill(obj_to_save, self.save_path)

    def _load_fit_obj_from_path(self, path):
        loaded_obj = util.load_dill(path)

        criterion = loaded_obj.criterion_class
        optimizer = loaded_obj.optimizer_class
        return SerializedPackagedModel(optimizer, criterion,
                                       loaded_obj.optim_params,
                                       loaded_obj.loader_creator)

    def check_saved(self):
        saved = os.path.isfile(self.save_path)
        return saved

    def _raise_not_saved(self):
        raise ValueError(f"manager_save_path: {self.save_path} doesn't exist")
