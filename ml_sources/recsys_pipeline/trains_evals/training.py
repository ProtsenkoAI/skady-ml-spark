from torch import nn
from torch import optim
import numpy as np


class Trainer:
    def __init__(self, lr=1e-4):
        self.lr = lr
        self.criterion = nn.MSELoss()

    def fit(self, assistant, dataset, nsteps=None, nepochs=None):
        loader = self.loader_builder.build(dataset)
        optimizer = self._create_optimizer(assistant)
        steps_left = self._get_steps_left(nsteps, nepochs, loader)
        batches_generator = self._create_batch_generator(loader, steps_left)
        for batch in batches_generator:
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

    def _create_batch_generator(self, dataset, steps):
        def generator(dataset=dataset, steps=steps):
            step_idx = 0
            while True:
                for batch in dataset:
                    yield batch
                    step_idx += 1
                    if step_idx >= steps:
                        return

        return generator()

    def _create_optimizer(self, assistant):
        weights = assistant.get_model().parameters()
        optimizer = optim.SparseAdam(list(weights), lr=self.lr)
        return optimizer

    def _get_steps_left(self, nsteps, nepochs, dataset):
        self._validate_fit_args(nsteps, nepochs)
        if not nsteps is None:
            return nsteps
        elif not nepochs is None:
            steps_in_epoch = len(dataset)
            return nepochs * steps_in_epoch

    def _validate_fit_args(self, steps, epochs):
        if steps is None and epochs is None:
            raise ValueError("Both steps and epochs are None")
        elif not steps is None and not epochs is None:
            raise ValueError("Have to specify steps or epochs")
