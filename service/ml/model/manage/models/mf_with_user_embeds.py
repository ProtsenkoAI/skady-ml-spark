from torch import nn
import torch
from typing import List


from ...expose import RecsysTorchModel
from .mf_net_component import MFNetComponent
from .emb_net_component import EmbNetComponent


class MFWithUserEmbeddings(RecsysTorchModel):
    # TODO: investigate best network architecture to deal with embeddings
    def __init__(self, nusers, nitems, mf_hidden_size, embeddings_path: str, embeddings_size: int):
        super().__init__()
        self.mf_comp = MFNetComponent(nusers, nitems, mf_hidden_size)
        self.embed_comp = EmbNetComponent(embeddings_path, embeddings_size)
        self.last_layer = nn.Linear(2, 1)

    def forward(self, users: torch.IntTensor, items: torch.IntTensor) -> torch.FloatTensor:
        mf_preds = self.mf_comp(users, items)
        print("before embed_preds")
        try:
            embed_preds = self.embed_comp(users, items)
        except KeyError as e:
            raise KeyError("Key error in embeddings component. Probably, user wasn't "
                           f"added to vectors db before creating the model, user: {e}").with_traceback(e.__traceback__)

        concat_features = torch.cat([mf_preds, embed_preds], dim=1)
        preds = self.last_layer(concat_features)
        return preds

    def add_users(self, nb_users: int):
        self.mf_comp.add_users(nb_users)
        self.embed_comp.add_users(nb_users)

    def add_items(self, nb_items: int):
        self.mf_comp.add_items(nb_items)
        self.embed_comp.add_items(nb_items)

    def delete_users(self, users_idxs: List[int]):
        self.mf_comp.delete_users(users_idxs)
        self.embed_comp.delete_users(users_idxs)

    def delete_items(self, items_idxs: List[int]):
        self.mf_comp.delete_items(items_idxs)
        self.embed_comp.delete_items(items_idxs)

    def prepare_to_save(self):
        self.embed_comp.prepare_to_save()

    def run_after_saving(self):
        self.embed_comp.run_after_saving()
