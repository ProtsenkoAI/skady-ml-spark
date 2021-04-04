from .types import User
from abc import ABC, abstractmethod


class Recommender(ABC):
    @abstractmethod
    def get_recommends(self, user: User):
        ...
