from abc import ABC, abstractmethod
from model_level import ModelManager


class Trainer(ABC):
    @abstractmethod
    def fit(self, manager: ModelManager, data):
        ...
