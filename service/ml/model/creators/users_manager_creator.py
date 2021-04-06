import os
from model.users_managing import SparkUsersManager
from model.expose.users_manager import UsersManager
from data.creators import ObtainerCreator
from vk.vk_user_embedding_obtainer import VkUserEmbeddingObtainer


class UsersManagerCreator:
    def __init__(self, config, common_params):
        if common_params["mode"] == "spark":
            # TODO: move to creator when it will be written
            worker_dir_path = os.path.join(common_params["paths"]["base_path"],
                                           common_params["paths"]["worker_dir"])
            obtainer_tasks_dir = os.path.join(worker_dir_path, common_params["paths"]["vk_obtainer_tasks_dir"])
            os.makedirs(worker_dir_path, exist_ok=True)
            os.makedirs(obtainer_tasks_dir, exist_ok=True)
            tasks_obtainer = ObtainerCreator(config, common_params).get_vk_embed_tasks_obtainer().get_data()
            vk_embed_obtainer = VkUserEmbeddingObtainer(tasks_obtainer, obtainer_tasks_dir)

            self.users_manager = SparkUsersManager(vk_embed_obtainer,
                                                   config["users_manager_params"]["users_add_delete_file_name"],
                                                   worker_dir_path=worker_dir_path)
        else:
            raise NotImplementedError

    def get(self) -> UsersManager:
        return self.users_manager
