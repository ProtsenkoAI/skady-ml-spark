import os
from model.users_managing import SparkUsersManager
from model.expose.users_manager import UsersManager


class UsersManagerCreator:
    def __init__(self, config, common_params):
        if common_params["mode"] == "spark":
            # TODO: move to creator when it will be written
            worker_dir_path = os.path.join(common_params["paths"]["base_path"],
                                           common_params["paths"]["worker_dir"])

            self.users_manager = SparkUsersManager(config["users_manager_params"]["users_add_delete_file_name"],
                                                   worker_dir_path=worker_dir_path)
        else:
            raise NotImplementedError

    def get(self) -> UsersManager:
        return self.users_manager
