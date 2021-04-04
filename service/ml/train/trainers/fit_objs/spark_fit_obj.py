from train.expose.fit_obj import FitObj
from model import Saver
from model import UsersManager
import pandas as pd
from model.expose.types import PackagedModel


class SparkFitObj(FitObj):
    def __init__(self, users_manager: UsersManager, saver: Saver):
        self.users_manager = users_manager
        self.saver = saver

    def __call__(self, batch: pd.DataFrame):
        manager, weights_updater, loader_builder = self.saver.load_packaged_model()
        self._add_and_delete_users_if_needed(manager)

        self.fit_on_batch(batch, manager, weights_updater, loader_builder)

        self.saver.save_packaged_model(PackagedModel(manager, weights_updater, loader_builder))

    def _add_and_delete_users_if_needed(self, manager):
        add_list = self.users_manager.get_add_list()
        del_list = self.users_manager.get_delete_list()
        for user in add_list:
            manager.add_user(user)
        for user in del_list:
            manager.delete_user(user)
        self.users_manager.clean_lists()
