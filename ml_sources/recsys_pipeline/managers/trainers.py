from torch import optim
from copy import copy
from sklearn import metrics
from torch import nn
import numpy as np


class Trainer:
    def __init__(self, model, dataset, preprocessor, lr=1e-4,
                 save_path="./model_weights.pt"):
        self.model = model
        self.preprocessor = preprocessor

        self.dataset = dataset
        self._reset_dataset_iter()
        self.lr = lr
        self.save_path = save_path
        self.steps_left = None

        self._init_train_parts()

    def _init_train_parts(self):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SparseAdam(list(self.model.parameters()), lr=self.lr)

    def get_model(self):
        return self.model

    def get_dataset_len(self):
        return len(self.dataset)

    def add_users(self, nusers):
        self.model.add_users(nusers)
        self._init_train_parts()
    
    def add_items(self, nitems):
        self.model.add_items(nitems)
        self._init_train_parts()
    
    # def get_recommends_for_users(self, users, item_ids):
    #     # TODO: separate into some ModelManager
    #     preds_by_user = []
    #     for user in users:
    #         pred = self.model(user, item_ids).squeeze().detach().numpy()
    #         recommended_items_idxs = np.argsort(pred)[::-1]
    #         preds_by_user.append(recommended_items_idxs)
    #     return preds_by_user

    def fit(self, nsteps=None, nepochs=None):
        steps_left = self._get_steps_left_from_steps_epochs(nsteps, nepochs)
        self._fit_while_steps_left(steps_left)

    def _fit_while_steps_left(self, steps_left):
        while steps_left:
            self._train_one_step()
            steps_left -= 1

    def _get_steps_left_from_steps_epochs(self, nsteps, nepochs):
        self._check_steps_epochs_correctness(nsteps, nepochs)
        if nsteps:
            return nsteps
        else:
            return nepochs * self.get_dataset_len()

    def _check_steps_epochs_correctness(self, nsteps, nepochs):
        both_none = nsteps is None and nepochs is None
        both_not_none = not nsteps is None and not nepochs is None
        if both_none or both_not_none:
            raise ValueError("Should specify steps or epochs, but entered params nsteps: {nsteps}, nepochs: {nepochs}")

    def _train_one_step(self):
        batch = self._get_next_batch()
        # self.update_model_with_new_users_and_items(batch)
        loss = self._calc_batch_loss(batch)
        self._optimize_weights(loss)

        return loss.item()

    def _calc_batch_loss(self, batch):
        features, labels = self.preprocessor.get_batch_tensor(batch)
        outputs = self.model(*features)
        return self.criterion(outputs.squeeze(), labels)

    def _optimize_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _get_next_batch(self):
        try:
            next_batch = next(self.dataset_iter)
        except (StopIteration, AttributeError) as e: # dataset ended or hadn't been initialized
            self._reset_dataset_iter()
            next_batch = next(self.dataset_iter)
        return next_batch

    def _reset_dataset_iter(self):
        self.dataset_iter = iter(self.dataset)
