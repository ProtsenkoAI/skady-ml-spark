from torch import optim
from torch import nn


class TorchWeightsUpdater:
    def __init__(self, optimizer, criterion):
        self.optimizer = optimizer
        self.criterion = criterion

    def fit_with_batch(self, model, batch):
        self._verify_ready_for_fit()
        loss = self._calc_batch_loss(batch, model)
        self._optimize_weights(loss)
        return loss.item()

    def _calc_batch_loss(self, batch, model):
        # TODO: use assistant
        features, labels = batch
        labels_proc = labels.float()
        users, items = features.split(split_size=1, dim=-1)
        preds = model(users.squeeze(1), items.squeeze(1))
        return self.criterion(preds.squeeze(), labels_proc)

    def _optimize_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _verify_ready_for_fit(self):
        assert not self.optimizer is None, "Have to prepare_for_fit() before fit_with_batch()"
