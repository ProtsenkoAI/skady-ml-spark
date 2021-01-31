import torch
from collections.abc import Iterable
from .preproc_helpers import wrap_in_list_if_number


class DataPreprocessor:
    def __init__(self, device):
        self.device = device

    def preprocess_batch(self, batch):
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
        # if isinstance(labels, torch.Tensor):
        #     if not isinstance(labels.dtype, float):
        #         labels = labels.float()
        # else:
        #     labels = torch.tensor(labels).to(self.device).float()
        # return labels

        # return torch.tensor(labels).to(self.device).float()
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        return labels.to(self.device).float()

    def _preproc_index_features(self, index_features):
        if not isinstance(index_features, torch.Tensor):
            index_features = torch.tensor(index_features)
        index_features = index_features.to(self.device).long()
        return index_features

    def _split_users_items(self, batch_or_pair):
        if isinstance(batch_or_pair[0], Iterable):
            users, items = zip(*batch_or_pair)
        else:
            users, items = batch_or_pair
        return users, items

