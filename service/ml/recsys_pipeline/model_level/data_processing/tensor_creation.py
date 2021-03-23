import torch


class TensorCreator:
    def __init__(self, device="cpu"):
        self.device = device

    def get_feature_tensor(self, feature):
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature)
        print("device", self.device)
        feature = feature.to(self.device).long()
        # feature = index_features.reshape(-1, 1)
        return feature

    def get_labels_tensor(self, labels):
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        return labels.to(self.device).float()
