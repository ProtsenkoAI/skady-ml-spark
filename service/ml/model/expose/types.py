from typing import NamedTuple
from data.expose import LoaderBuilder
from train.expose import WeightsUpdater


User = int

PackagedModel = NamedTuple("PackagedModel", [("updater", WeightsUpdater),
                                             ("loader_builder", LoaderBuilder)]
                           )
