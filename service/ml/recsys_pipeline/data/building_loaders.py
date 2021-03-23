from torch.utils import data as torch_data
import pandas as pd
from . import datasets


class StandardLoaderBuilder:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build(self, interacts):
        dataset = datasets.InteractDataset(interacts)
        return torch_data.DataLoader(dataset, batch_size=self.batch_size)


class UserItemsLoaderBuilder:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.user_colname = "user_actor_id"
        self.item_colname = "user_proposed_id"

    def build(self, user, items):
        interacts = self._create_interacts(user, items)
        dataset = datasets.InteractDataset(interacts, has_label=False)
        loader = torch_data.DataLoader(dataset, batch_size=self.batch_size)
        return loader

    def _create_interacts(self, user, items):
        interacts = pd.DataFrame({self.item_colname: items}, columns=[self.user_colname, self.item_colname])
        interacts[self.user_colname] = user
        return interacts