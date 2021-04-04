from abc import ABC, abstractmethod
from model import ModelManager
import pandas as pd


class WeightsUpdater(ABC):
    DataBatch = pd.DataFrame

    @abstractmethod
    def get_optimizer_class(self):
        ...

    @abstractmethod
    def get_criterion_class(self):
        ...

    def get_optim_params(self):
        ...

    @abstractmethod
    def prepare_for_fit(self, model_manager: ModelManager):
        ...

    @abstractmethod
    def fit_with_batch(self, model_manager: ModelManager, batch: DataBatch):
        ...
