from train_eval.updating_weights.weights_updater import WeightsUpdater
from train_eval.trainers.trainer import Trainer


class SimpleTrainer(Trainer):
    def __init__(self, loader_builder, lr=1e-4):
        self.loader_builder = loader_builder
        self.weights_updater = WeightsUpdater(lr)

    def fit(self, manager, interacts, max_epoch=None, max_step=None):
        # TODO: switch from interacts to obtainer
        manager.update_with_interacts(interacts)
        self.weights_updater.prepare_for_fit(manager)
        loader = self.loader_builder.build(interacts)
        step_cnt = 0
        epoch_cnt = 0
        while True:
            for batch in loader:
                if self._check_stop_cond(step_cnt, epoch_cnt, max_step, max_epoch):
                    return
                self.weights_updater.fit_with_batch(manager, batch)
                step_cnt += 1
            epoch_cnt += 1

    def _check_stop_cond(self, cur_step, cur_epoch, max_step=None, max_epoch=None):
        if not max_step is None:
            if max_step <= cur_step:
                return True
        if not max_epoch is None:
            if max_epoch <= cur_epoch:
                return True
        return False
