from app.main.service.data_managers import GroupsDataManager, MatrixDataManager, UsersIdsDataManager
import vk_api


class BaseUserManipulator:
    def __init__(self, users_ids_manager: UsersIdsDataManager,
                 matrix_manager: MatrixDataManager, groups_manager: GroupsDataManager):
        """

        :param users_ids_manager: implements get_ids() and save_new_ids()
        :param matrix_manager: implements get_matrix() and save_new_matrix()
        """
        self.users_ids_manager = users_ids_manager
        self.matrix_manager = matrix_manager
        self.groups_manager = groups_manager
        self.default_creds = ["+79898797278", "63pJnPGj1"]

    def create_session(self, token):
        if token is None:
            return vk_api.VkApi(*self.default_creds)
        else:
            return vk_api.VkApi(token=token)
