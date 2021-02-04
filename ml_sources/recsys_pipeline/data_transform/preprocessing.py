import torch
from .preproc_helpers import wrap_in_list_if_number


class TensorCreator:
    def __init__(self, device):
        self.device = device

    @wrap_in_list_if_number
    def get_users_tensor(self, users):
        return self._preproc_index_features(users)
    
    @wrap_in_list_if_number
    def get_items_tensor(self, items):
        return self._preproc_index_features(items)

    @wrap_in_list_if_number
    def get_labels_tensor(self, labels):
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        return labels.to(self.device).float()

    def _preproc_index_features(self, index_features):
        if not isinstance(index_features, torch.Tensor):
            index_features = torch.tensor(index_features)
        index_features = index_features.to(self.device).long()
        # index_features = index_features.reshape(-1, 1)
        return index_features
