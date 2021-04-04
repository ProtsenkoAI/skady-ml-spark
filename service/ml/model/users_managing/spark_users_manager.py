import os
import json
from ..expose.users_manager import UsersManager
from ..expose.types import User


class SparkUsersManager(UsersManager):
    def __init__(self, users_add_delete_file_name: str, worker_dir_path: str):
        self.lists_file_path = os.path.join(worker_dir_path, users_add_delete_file_name)
        self._create_lists_file_if_need()

    def clean_lists(self):
        cont = self._get_file_content()
        cont["add"] = []
        cont["delete"] = []
        print("saving clean lists")
        self._save_cont(cont)

    def add_user(self, user: User):
        list_of_users_to_add = self.get_add_list()
        list_of_users_to_add.append(user)
        self._save_add_list(list_of_users_to_add)

    def delete_user(self, user: User):
        list_of_users_to_delete = self.get_delete_list()
        list_of_users_to_delete.append(user)
        self._save_delete_list(list_of_users_to_delete)

    def get_add_list(self):
        return self._get_file_content()["add"]

    def get_delete_list(self):
        return self._get_file_content()["delete"]

    def _get_file_content(self) -> dict:
        with open(self.lists_file_path) as f:
            cont = json.load(f)
        return cont

    def _create_lists_file_if_need(self):
        if not os.path.isfile(self.lists_file_path):
            lists_to_add_and_delete = {"add": [],
                                       "delete": []}
            with open(self.lists_file_path, "w") as f:
                json.dump(lists_to_add_and_delete, f)

    def _save_add_list(self, add_list):
        cont = self._get_file_content()
        cont["add"] = add_list
        self._save_cont(cont)

    def _save_delete_list(self, delete_list):
        cont = self._get_file_content()
        cont["delete"] = delete_list
        self._save_cont(cont)

    def _save_cont(self, cont: dict):
        with open(self.lists_file_path, "w") as f:
            json.dump(cont, f)
