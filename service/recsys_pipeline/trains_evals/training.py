from torch import nn
from torch import optim


class Trainer:
    def __init__(self, lr=1e-4):
        self.lr = lr
        self.criterion = nn.MSELoss()

    def fit(self, assistant, loader):
        optimizer = self._create_optimizer(assistant)
        for batch in loader:
            self._train_one_step(batch, assistant, optimizer)

    def _train_one_step(self, batch, assistant, optimizer):
        loss = self._calc_batch_loss(batch, assistant)
        self._optimize_weights(loss, optimizer)
        return loss.item()

    def _calc_batch_loss(self, batch, assistant):
        features, labels = batch
        # print("features", features)
        preds = assistant.preproc_then_forward(features)
        labels_proc = assistant.preproc_labels(labels)
        return self.criterion(preds.squeeze(), labels_proc)

    def _optimize_weights(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _create_optimizer(self, assistant):
        weights = assistant.get_model().parameters()
        optimizer = optim.SparseAdam(list(weights), lr=self.lr)
        return optimizer
