from abc import ABC, abstractmethod


class StreamingTrainer(ABC):
    @abstractmethod
    def fit(self, data_manager):
        ...

    @abstractmethod
    def stop_fit(self):
        ...

    @abstractmethod
    def await_fit(self, timeout=None):
        ...
