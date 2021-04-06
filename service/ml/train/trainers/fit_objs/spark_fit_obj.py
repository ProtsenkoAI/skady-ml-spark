from train.expose.fit_obj import FitObj
from model import Saver
from model import UsersManager
import pandas as pd
from model.expose.types import PackagedModel


class SparkFitObj(FitObj):
    def __init__(self, users_manager: UsersManager, packaged_train_saver: Saver, model_manager_class, manager_saver):
        self.users_manager = users_manager
        self.packaged_train_saver = packaged_train_saver
        self.model_manager_class = model_manager_class
        self.manager_saver = manager_saver

    def __call__(self, batch: pd.DataFrame):
        manager = self.model_manager_class.load(self.manager_saver)
        weights_updater, loader_builder = self.packaged_train_saver.load_packaged_model()
        self._add_and_delete_users_if_needed(manager)

        self.fit_on_batch(batch, manager, weights_updater, loader_builder)

        self.packaged_train_saver.save_packaged_model(PackagedModel(weights_updater, loader_builder))
        manager.save()

    def _add_and_delete_users_if_needed(self, manager):
        add_list = self.users_manager.get_add_list()
        del_list = self.users_manager.get_delete_list()
        for user in add_list:
            manager.add_user(user)
        for user in del_list:
            manager.delete_user(user)
        self.users_manager.clean_lists()
