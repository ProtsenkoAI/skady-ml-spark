from app.main.service.manipulators.base_user_manipulator import BaseUserManipulator


class UserDeleter(BaseUserManipulator):
    def delete_user(self, user_id, access_token):
        self.create_session(access_token)
        self.matrix_manager.delete_row_and_col(user_id)
        self.groups_manager.remove(user_id)