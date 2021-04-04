from abc import ABC, abstractmethod
from typing import List
from model.expose.types import User


class RecommendsPostprocessor(ABC):
    @abstractmethod
    def process(self, user: User, raw_recommends: List[User]) -> List[User]:
        r"""Does any filtering/reordering of recommends it needs to do based on business rules
        (for example, filter users that were already shown, are too far away and so on."""
        ...
