from torch import optim
from torch import nn


class WeightsUpdater:
    def __init__(self, lr):
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = None

    def prepare_for_fit(self, assistant):
        weights = assistant.get_model().parameters()
        optimizer = optim.SparseAdam(list(weights), lr=self.lr)
        self.optimizer = optimizer

    def fit_with_batch(self, assistant, batch):
        self._verify_ready_for_fit()
        loss = self._calc_batch_loss(batch, assistant)
        self._optimize_weights(loss)
        return loss.item()

    def _calc_batch_loss(self, batch, assistant):
        features, labels = batch
        preds, labels_proc = assistant.preproc_forward(features, labels=labels)
        return self.criterion(preds.squeeze(), labels_proc)

    def _optimize_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _verify_ready_for_fit(self):
        assert not self.optimizer is None, "Have to prepare_for_fit() before fit_with_batch()"
