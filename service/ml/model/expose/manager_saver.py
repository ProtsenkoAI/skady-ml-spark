from abc import abstractmethod, ABC
from typing import Tuple
from . import RecsysTorchModel
from . import Processor


class ManagerSaver(ABC):
    @abstractmethod
    def load(self) -> Tuple[RecsysTorchModel, Processor]:
        ...

    @abstractmethod
    def save(self, model: RecsysTorchModel, processor: Processor):
        ...
