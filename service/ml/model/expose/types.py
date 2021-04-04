from typing import NamedTuple
from .model_manager import ModelManager
from data.expose import LoaderBuilder
from train.expose import WeightsUpdater


User = int

PackagedModel = NamedTuple("PackagedModel", [("manager", ModelManager),
                                             ("updater", WeightsUpdater),
                                             ("loader_builder", LoaderBuilder)]
                           )