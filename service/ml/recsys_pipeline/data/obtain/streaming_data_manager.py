from typing import Callable
from abc import ABC, abstractmethod
from .fit_data_manager import FitDataManager


class StreamingDataManager(FitDataManager, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def apply_whenever_data_recieved(self, func: Callable):
        ...
