from torch import optim
from torch import nn


class WeightsUpdater:
    def __init__(self, lr=1e-4, criterion_class=nn.MSELoss, optimizer_class=optim.SparseAdam):
        self.lr = lr
        self.criterion_class = criterion_class
        self.criterion = criterion_class()
        self.optimizer_class = optimizer_class
        self.optimizer = None

    def get_optimizer_class(self):
        return self.optimizer_class

    def get_criterion_class(self):
        return self.criterion_class

    def get_optim_params(self) -> dict:
        return {"lr": self.lr}

    def prepare_for_fit(self, model_manager):
        weights = model_manager.get_model().parameters()
        optimizer = self.optimizer_class(list(weights), lr=self.lr)
        self.optimizer = optimizer

    def fit_with_batch(self, assistant, batch):
        self._verify_ready_for_fit()
        loss = self._calc_batch_loss(batch, assistant)
        self._optimize_weights(loss)
        return loss.item()

    def _calc_batch_loss(self, batch, model_manager):
        # TODO: move splitting to processor
        features, labels = batch
        preds, labels_proc = model_manager.preproc_forward(features, labels=labels)
        return self.criterion(preds.squeeze(), labels_proc)

    def _optimize_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _verify_ready_for_fit(self):
        assert not self.optimizer is None, "Have to prepare_for_fit() before fit_with_batch()"
