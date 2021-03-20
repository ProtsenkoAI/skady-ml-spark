import os
import codecs
import dill
from time import time

from typing import Tuple
from model_level.model_managing import ModelManager
from train_eval.training.weights_updater import WeightsUpdater


class SparkSaver:
    def __init__(self, save_path):
        self.save_path = save_path

    def load(self) -> Tuple[ModelManager, WeightsUpdater]:
        raise NotImplementedError

    def save(self, manager: ModelManager, updater: WeightsUpdater):
        raise NotImplementedError

    # def _load_fit_obj(self) -> ModelObj:
    #     if os.path.isfile(self.manager_save_path):
    #         return self._load_fit_obj_from_path(self.manager_save_path)
    #     else:
    #         raise ValueError(f"manager_save_path: {self.manager_save_path} doesn't exist")
    #
    # def _create_fit_obj(self, model, criterion, optimizer, opt_params) -> SerializablePackagedModel:
    #     return SerializablePackagedModel(
    #             model=model,
    #             criterion=criterion,
    #             optimizer_class=optimizer,
    #             optim_params=opt_params
    #     )
    #
    # def _save_fit_obj(self, fit_obj: SerializablePackagedModel):
    #     start = time()
    #     os.makedirs(os.path.dirname(self.manager_save_path), exist_ok=True)
    #     serialized = dill.dumps(fit_obj)
    #     with open(self.manager_save_path, "wb") as f:
    #         encoded = codecs.encode(serialized, "base64")
    #         f.write(encoded)
    #     print("time to save", time() - start)
    #
    # def _load_fit_obj_from_path(self, path):
    #     start = time()
    #     with codecs.open(path, "rb") as f:
    #         torch_obj_decoded = codecs.decode(f.read(), "base64")
    #     loaded_obj = dill.loads(torch_obj_decoded)
    #     model, criterion = loaded_obj.model, loaded_obj.criterion
    #     optimizer = loaded_obj.optimizer_class(model.parameters(), **loaded_obj.optim_params)
    #     print("time to load", time() - start)
    #     return ModelObj(model, optimizer, criterion, loaded_obj.optim_params)
