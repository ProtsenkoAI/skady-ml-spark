from train_eval.trainers.trainer import Trainer
from abc import ABC, abstractmethod
from typing import Union
from model_level.model_managing import ModelManager
from data.obtain.data_managers import StreamingDataManager
from train_eval.trainers.fit_objs.fit_obj import FitObj


class StreamingTrainer(Trainer, ABC):
    # TODO: move everything that could be moved from spark trainer

    def __init__(self):
        self.data_manager: Union[StreamingDataManager, None] = None

    def fit(self, manager: ModelManager, data_manager: StreamingDataManager, **fit_kwargs):
        self.data_manager = data_manager
        fit_obj = self.get_fit_obj(manager)
        data_manager.apply_to_each_batch(fit_obj, kwargs=fit_kwargs)

    @abstractmethod
    def get_fit_obj(self, manager) -> FitObj:
        ...

    def start_fitting(self, manager: ModelManager, fit_obj: FitObj):
        """does nothing by default"""
        pass

    def stop_fit(self):
        self.data_manager.stop_queries()

    def await_fit(self, timeout=None):
        self.data_manager.await_all_queries(timeout)
