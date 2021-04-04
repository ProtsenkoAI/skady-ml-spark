import torch
from .mf_net_component import MFNetComponent
from ...expose import RecsysTorchModel
from typing import List


class MFWithBiasModel(RecsysTorchModel):
    def __init__(self, nusers, nitems, hidden_size):
        super().__init__()
        self.nusers = nusers
        self.nitems = nitems
        self.hidden_size = hidden_size
        self.mf_comp = MFNetComponent(nusers, nitems, hidden_size)

    def get_init_kwargs(self):
        return {"nusers": self.nusers, "nitems": self.nitems, "hidden_size": self.hidden_size}

    def forward(self, users: torch.IntTensor, items: torch.IntTensor) -> torch.FloatTensor:
        return self.mf_comp(users, items)

    def add_users(self, nb_users):
        self.mf_comp.add_users(nb_users)

    def add_items(self, nb_items):
        self.mf_comp.add_items(nb_items)

    def delete_users(self, users_idxs: List[int]):
        self.mf_comp.delete_users(*users_idxs)

    def delete_items(self, items_idxs: List[int]):
        self.mf_comp.delete_items(*items_idxs)
