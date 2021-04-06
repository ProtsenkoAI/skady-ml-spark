from typing import Union
from data import StreamingDataManager
from ..expose import FitObj, StreamingTrainer
from model.expose.types import PackagedModel


class StreamingTrainerImpl(StreamingTrainer):
    def __init__(self, fit_obj: FitObj, saver, weights_updater, loader_builder, manager_creator):
        self.data_manager: Union[StreamingDataManager, None] = None
        self.fit_obj = fit_obj

        # objs needed to save model at the start
        self.saver = saver
        self.weights_updater = weights_updater
        self.loader_builder = loader_builder
        self.manager_creator = manager_creator

    def fit(self, data_manager: StreamingDataManager):
        self.data_manager = data_manager
        self._create_new_model_if_needed()
        data_manager.apply_to_each_batch(self.fit_obj)

    def _create_new_model_if_needed(self):
        if not self.saver.check_saved():
            # TODO: investigate: is it ok to use creator in the implementation part???
            manager = self.manager_creator.get()
            manager.save()
            packaged = PackagedModel(self.weights_updater, self.loader_builder)
            self.saver.save_packaged_model(packaged)
            print("saved packaged")

    def stop_fit(self):
        self.data_manager.stop_queries()

    def await_fit(self, timeout=None):
        self.data_manager.await_all_queries(timeout)
