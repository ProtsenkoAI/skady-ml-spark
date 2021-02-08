from torch.utils import data as torch_data
import itertools
import pandas as pd
from . import datasets


class StandardLoaderBuilder:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build(self, interacts):
        dataset = datasets.InteractDataset(interacts)
        return torch_data.DataLoader(dataset, batch_size=self.batch_size)


# class UsersItemsLoaderBuilder:
#     def __init__(self, batch_size):
#         self.batch_size = batch_size
#         self.user_colname = "user_id"
#         self.item_colname = "anime_id"
#
#     def build(self, users, items):
#         interacts = self._create_interacts(users, items)
#         dataset = datasets.InteractDataset(interacts, has_label=False)
#         loader = torch_data.DataLoader(dataset, batch_size=self.batch_size)
#         return loader
#
#     def _create_interacts(self, users, items):
#         all_pairs = list(itertools.product(users, items))
#         interacts = pd.DataFrame(all_pairs, columns=[self.user_colname, self.item_colname])
#         return interacts