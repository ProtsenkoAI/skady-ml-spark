# TODO: replace pd.DataFrame with some simpler object
import pandas as pd
from abc import ABC, abstractmethod


class FitObj(ABC):
    @abstractmethod
    def __call__(self, batch: pd.DataFrame):
        ...

    @staticmethod
    def fit_on_batch(batch, manager, weights_updater, loader_builder):
        weights_updater.prepare_for_fit(manager)
        batch_iterator = loader_builder.build(batch)
        for minibatch in batch_iterator:
            weights_updater.fit_with_batch(manager, minibatch)
