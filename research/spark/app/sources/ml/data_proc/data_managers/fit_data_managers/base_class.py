from typing import Tuple


from abc import ABCMeta, abstractmethod
from ....spark.consts import Types


class FitDataManager(metaclass=ABCMeta):
    @abstractmethod
    def union_with_saved_unused(self, new_data: Types.RawData) -> Types.RawData:
        ...

    @abstractmethod
    def prepare_for_fit(self, raw_fit_data: Types.RawData) -> Types.FitData:
        ...

    @abstractmethod
    def split_data_to_fit_and_unused(self, joined_data: Types.RawData) -> Tuple[Types.RawData, Types.RawData]:
        ...

    @abstractmethod
    def save_unused(self, unused_data: Types.RawData):
        ...
