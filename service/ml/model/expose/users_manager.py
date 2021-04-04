from abc import ABC, abstractmethod
from .types import User


class UsersManager(ABC):
    @abstractmethod
    def clean_lists(self):
        ...

    @abstractmethod
    def add_user(self, user: User):
        ...

    @abstractmethod
    def delete_user(self, user: User):
        ...

    @abstractmethod
    def get_add_list(self):
        ...

    @abstractmethod
    def get_delete_list(self):
        ...
