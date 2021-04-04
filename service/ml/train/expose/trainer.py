from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def fit(self, data_manager):
        ...
