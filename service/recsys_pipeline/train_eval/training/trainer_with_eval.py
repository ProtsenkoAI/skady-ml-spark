import numpy as np
from sklearn import model_selection
from .weights_updater import WeightsUpdater
from model_level.assistance import ModelAssistant


class EvalTrainer:
    def __init__(self, validator, loader_builder, saver, test_percentage=0.2, steps_between_evals=200,
                 lr=1e-4, save_best_model=True):
        self.validator = validator
        self.loader_builder = loader_builder
        self.saver = saver
        self.test_percent = test_percentage
        self.steps_between_evals = steps_between_evals
        self.weights_updater = WeightsUpdater(lr=lr)
        self.val_results = []
        self.saved_checkpoint = None
        self.save_best_model = save_best_model

    def fit(self, assistant, interacts, max_epoch=None, max_step=None, stop_patience=None):
        assistant.update_with_interacts(interacts)
        self.weights_updater.prepare_for_fit(assistant)
        train_interacts, val_interacts = self._split_train_test(interacts)
        train_loader = self.loader_builder.build(train_interacts)

        step_cnt = 0
        epoch_cnt = 0
        while True:
            for batch in train_loader:
                if self._check_stop_cond(step_cnt, epoch_cnt, max_step, max_epoch, stop_patience):
                    return
                if (step_cnt + 1) % self.steps_between_evals == 0:
                    self._eval(assistant, val_interacts)

                self.weights_updater.fit_with_batch(assistant, batch)

                step_cnt += 1
            epoch_cnt += 1

    def _eval(self, assistant, inters):
        score = self.validator.evaluate(assistant, inters)
        self.val_results.append(score)
        if self._check_has_to_save():
            self.saved_checkpoint = assistant.save(self.saver)

    def get_val_results(self):
        return self.val_results

    def load_best(self):
        assert not self.saved_checkpoint is None, "The trainer hadn't made any checkpoints yet"
        assistant = ModelAssistant.from_save(self.saver, self.saved_checkpoint)
        return assistant

    def _check_stop_cond(self, cur_step, cur_epoch, max_step=None, max_epoch=None, stop_patience=5):
        if not max_step is None:
            if max_step <= cur_step:
                return True
        if not max_epoch is None:
            if max_epoch <= cur_epoch:
                return True
        if self._stop_patience_exceeded(self.val_results, stop_patience):
            return True
        return False

    def _stop_patience_exceeded(self, eval_res, patience=None):
        if len(eval_res) != 0 and not patience is None:
            steps_passed = len(eval_res)
            best_result_step = np.argmax(eval_res)
            steps_after_last_best_val = (steps_passed - best_result_step - 1)
            if steps_after_last_best_val >= patience:
                print("EARLY STOPPING", "eval_vals", eval_res, "best_result_step", best_result_step)
                return True
        return False

    def _check_has_to_save(self):
        if not self.save_best_model:
            return False
        last_is_best = np.argmax(self.val_results) == (len(self.val_results) - 1)
        return last_is_best

    def _split_train_test(self, interacts):
        return model_selection.train_test_split(interacts, test_size=self.test_percent)
