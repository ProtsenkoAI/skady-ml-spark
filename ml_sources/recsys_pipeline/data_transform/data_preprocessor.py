import torch
from collections.abc import Iterable
from .preproc_helpers import wrap_in_list_if_number

class DataPreprocessor:
    def __init__(self, device):
        self.device = device

    def preprocess_batch(self, batch):
        # (users, items), labels = self._split_batch(batch)
        features, labels = batch
        users, items = self._split_users_items(features)

        users_proc = self.preprocess_users(users)
        items_proc = self.preprocess_items(items)
        labels_preprocessed = self.preprocess_labels(labels)
        return (users_proc, items_proc), labels_preprocessed

    def preprocess_x(self, users_items):
        users, items = self._split_users_items(users_items)
        users_proc = self.preprocess_users(users)
        items_proc = self.preprocess_items(items)
        
        return users_proc, items_proc

    @wrap_in_list_if_number
    def preprocess_users(self, users):
        return self._preproc_index_features(users)
    
    @wrap_in_list_if_number
    def preprocess_items(self, items):
        return self._preproc_index_features(items)

    @wrap_in_list_if_number
    def preprocess_labels(self, labels):
        return torch.tensor(labels).to(self.device).float()

    def _preproc_index_features(self, index_features):
        return torch.tensor(index_features, dtype=torch.long, device=self.device)

    def _split_batch(self, batch):
        print("batch in _split_batch", batch)
        print("*batch", *batch)
        print(len(batch))
        features, labels = zip(*batch)
        features_proc = self._split_users_items(features)
        return features_proc, labels

    def _split_users_items(self, batch_or_pair):
        if isinstance(batch_or_pair[0], Iterable):
            users, items = zip(*batch_or_pair)
        else:
            users, items = batch_or_pair
        return users, items

