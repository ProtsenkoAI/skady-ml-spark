from abc import ABC, abstractmethod


class Obtainer(ABC):
    @abstractmethod
    def get_data(self):
        ...
