from abc import ABC, abstractmethod
from .types import PackagedModel


class Saver(ABC):
    @abstractmethod
    def check_saved(self) -> bool:
        ...

    @abstractmethod
    def save_packaged_model(self, packaged_model: PackagedModel):
        ...

    @abstractmethod
    def load_packaged_model(self) -> PackagedModel:
        ...
