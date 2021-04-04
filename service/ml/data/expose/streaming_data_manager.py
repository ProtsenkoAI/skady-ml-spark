from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
import pandas as pd


class StreamingDataManager(ABC):
    DataBatch = pd.DataFrame

    @abstractmethod
    def stop_queries(self):
        ...

    @abstractmethod
    def await_all_queries(self, timeout: Optional[int] = None):
        ...

    @abstractmethod
    def apply_to_each_batch(self, func: Callable[[DataBatch], Any], kwargs: Optional[dict] = None):
        ...

    @abstractmethod
    def apply_to_each_row(self, func: Callable[[DataBatch], Any], kwargs: Optional[dict] = None):
        ...
