import os
import codecs
import dill

from typing import Tuple
from collections import namedtuple
from model_level.model_managing import ModelManager
from train_eval.updating_weights.weights_updater import WeightsUpdater


SerializablePackagedModel = namedtuple("SerializablePackagedModel",
                                       ["model", "optimizer_class", "criterion_class", "optim_params", "processor",
                                        "loader_creator"])

ModelObj = namedtuple("ModelObj", ["model", "optimizer_class", "criterion_class", "optim_params", "processor",
                                   "loader_creator"])


class SparkSaver:
    # TODO: refactor saving flow
    def __init__(self, save_path):
        self.save_path = save_path

    def load(self) -> Tuple[ModelManager, WeightsUpdater]:
        if os.path.isfile(self.save_path):
            model_obj = self._load_fit_obj_from_path(self.save_path)
            manager = ModelManager(model_obj.model, model_obj.processor)
            weights_updater = WeightsUpdater(criterion_class=model_obj.criterion_class,
                                             optimizer_class=model_obj.optimizer_class,
                                             **model_obj.optim_params)
            loader_builder = model_obj.loader_creator
            return manager, weights_updater, loader_builder
        else:
            raise ValueError(f"manager_save_path: {self.manager_save_path} doesn't exist")

    def save(self, manager: ModelManager, updater: WeightsUpdater, loader_creator):
        model = manager.get_model()
        processor = manager.get_processor()
        optimizer = updater.get_optimizer_class()
        criterion = updater.get_criterion_class()
        optim_params = updater.get_optim_params()
        obj_to_save = SerializablePackagedModel(
                model=model,
                processor=processor,
                criterion_class=criterion,
                optimizer_class=optimizer,
                optim_params=optim_params,
                loader_creator=loader_creator
        )

        self._dump(obj_to_save)

    def _dump(self, fit_obj: SerializablePackagedModel):
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
        return ModelObj(model, optimizer, criterion, loaded_obj.optim_params, processor, loaded_obj.loader_creator)
