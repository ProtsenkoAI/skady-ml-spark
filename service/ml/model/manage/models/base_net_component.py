from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import List


class BaseNetComponent(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, users: torch.IntTensor, items: torch.IntTensor) -> torch.FloatTensor:
        ...

    @abstractmethod
    def add_users(self, nb_users: int):
        ...

    @abstractmethod
    def add_items(self, nb_items: int):
        ...

    @abstractmethod
    def delete_users(self, users_idxs: List[int]):
        ...

    @abstractmethod
    def delete_items(self, items_idxs: List[int]):
        ...
