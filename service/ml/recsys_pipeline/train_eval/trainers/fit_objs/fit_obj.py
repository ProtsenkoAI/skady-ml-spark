# TODO: replace pd.DataFrame with some simpler object
import pandas as pd
from abc import ABC, abstractmethod


class FitObj(ABC):
    @abstractmethod
    def __call__(self, batch: pd.DataFrame, **kwargs):
        ...
