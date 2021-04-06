from .base_net_component import BaseNetComponent
from torch import nn
import torch
from typing import List
import shelve


class EmbNetComponent(BaseNetComponent):
    # TODO: investigate best practices to store, update and load vectors for users (embeddings)
    #   because now it's stored in the shelve file and with high load the solution can become too slow
    def __init__(self, embeddings_path: str, embeddings_size: int):
        super().__init__()
        self.embeddings_path = embeddings_path
        self.embeddings_dict = self._load_users_embeds(embeddings_path)
        self.linear_layer = nn.Linear(2 * embeddings_size, 1)

    def _load_users_embeds(self, path: str) -> dict:
        db = shelve.open(path)
        db_in_py_dict = dict(db)
        db.close()
        return db_in_py_dict

    def forward(self, users: torch.IntTensor, items: torch.IntTensor) -> torch.FloatTensor:
        users_embeds = self._get_embeds(users)
        items_embeds = self._get_embeds(items)
        concat_embeds = torch.cat([users_embeds, items_embeds], dim=1)
        linear_out = self.linear_layer(concat_embeds)
        return linear_out

    def _get_embeds(self, tensor: torch.Tensor):
        device = torch.device("cuda" if tensor.is_cuda else "cpu")
        out_lst = []
        for user_id in torch.squeeze(tensor):
            embed = self.embeddings_dict[int(user_id)]
            out_lst.append(embed)
        tensor = torch.tensor(out_lst, device=device)
        return tensor

    # functions just do nothing because users' embeddings are deleted on embeddings side
    def add_users(self, nb_users: int):
        ...

    def add_items(self, nb_items: int):
        ...

    def delete_users(self, users_idxs: List[int]):
        ...

    def delete_items(self, items_idxs: List[int]):
        ...

    def prepare_to_save(self):
        del self.embeddings_dict

    def run_after_saving(self):
        self.embeddings_dict = self._load_users_embeds(self.embeddings_path)
