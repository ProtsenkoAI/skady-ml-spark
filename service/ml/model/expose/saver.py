from abc import ABC, abstractmethod
from .model_manager import ModelManager
from .types import PackagedModel


class Saver(ABC):
    @abstractmethod
    def check_saved(self) -> bool:
        ...

    @abstractmethod
    def save_model_manager(self, model_manager: ModelManager):
        ...

    @abstractmethod
    def load_model_manager(self) -> ModelManager:
        ...

    @abstractmethod
    def save_packaged_model(self, packaged_model: PackagedModel):
        ...

    @abstractmethod
    def load_packaged_model(self) -> PackagedModel:
        ...
