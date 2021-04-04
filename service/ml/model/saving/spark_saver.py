# TODO: check for better ways for serializing strategies (refactor)
# TODO: check that serialises well
# TODO: test time to load and save model
import os
import codecs
import dill
from train.updating_weights.weights_updater_impl import WeightsUpdaterImpl
from .. import Saver, ModelManager
from ..expose.types import PackagedModel
from collections import namedtuple

SerializedPackagedModel = namedtuple("SerializablePackagedModel",
                                     ["model", "optimizer_class", "criterion_class", "optim_params", "processor",
                                      "loader_creator"])


class SparkSaver(Saver):
    # TODO: not to hardcode classes
    def __init__(self, save_path):
        self.save_path = save_path

    def load_packaged_model(self) -> PackagedModel:
        if not self.check_saved():
            self._raise_not_saved()
        model_obj = self._load_fit_obj_from_path(self.save_path)
        manager = ModelManager(model_obj.model, model_obj.processor)
        weights_updater = WeightsUpdaterImpl(criterion_class=model_obj.criterion_class,
                                             optimizer_class=model_obj.optimizer_class,
                                             **model_obj.optim_params)
        loader_builder = model_obj.loader_creator
        return PackagedModel(manager, weights_updater, loader_builder)

    def load_model_manager(self):
        if not self.check_saved():
            self._raise_not_saved()
        model_obj = self._load_fit_obj_from_path(self.save_path)
        manager = ModelManager(model_obj.model, model_obj.processor)
        return manager

    def save_model_manager(self, manager: ModelManager):
        old_manager, weights_updater, loader_builder = self.load_packaged_model()
        del old_manager
        packaged = PackagedModel(manager, weights_updater, loader_builder)
        self.save_packaged_model(packaged)

    def save_packaged_model(self, packaged_model: PackagedModel):
        manager, updater, loader_creator = packaged_model.manager, packaged_model.updater, packaged_model.loader_builder
        model = manager.get_model()
        processor = manager.get_processor()
        optimizer = updater.get_optimizer_class()
        criterion = updater.get_criterion_class()
        optim_params = updater.get_optim_params()
        obj_to_save = SerializedPackagedModel(
            model=model,
            processor=processor,
            criterion_class=criterion,
            optimizer_class=optimizer,
            optim_params=optim_params,
            loader_creator=loader_creator
        )

        self._dump(obj_to_save)

    def _dump(self, fit_obj: SerializedPackagedModel):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        serialized = dill.dumps(fit_obj)
        with open(self.save_path, "wb") as f:
            encoded = codecs.encode(serialized, "base64")
            f.write(encoded)

    def _load_fit_obj_from_path(self, path):
        with codecs.open(path, "rb") as f:
            torch_obj_decoded = codecs.decode(f.read(), "base64")
        loaded_obj = dill.loads(torch_obj_decoded)
        model, criterion, processor = loaded_obj.model, loaded_obj.criterion_class, loaded_obj.processor
        optimizer = loaded_obj.optimizer_class
        return SerializedPackagedModel(model, optimizer, criterion,
                                       loaded_obj.optim_params, processor,
                                       loaded_obj.loader_creator)

    def check_saved(self):
        saved = os.path.isfile(self.save_path)
        return saved

    def _raise_not_saved(self):
        raise ValueError(f"manager_save_path: {self.save_path} doesn't exist")