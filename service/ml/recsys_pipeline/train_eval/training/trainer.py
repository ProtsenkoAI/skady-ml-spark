from abc import ABC, abstractmethod
from model_level import ModelManager
from data.obtain import FitDataManager


class Trainer(ABC):
    @abstractmethod
    def fit(self, manager: ModelManager, data: FitDataManager):
        ...
