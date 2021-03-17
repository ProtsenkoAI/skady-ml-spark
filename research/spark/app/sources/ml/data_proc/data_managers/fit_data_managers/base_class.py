from typing import Tuple


from abc import ABCMeta, abstractmethod
from ....spark.consts import Types


class FitDataManager(metaclass=ABCMeta):
    @abstractmethod
    def prepare_for_fit(self, raw_fit_data: Types.RawData) -> Types.FitData:
        ...
