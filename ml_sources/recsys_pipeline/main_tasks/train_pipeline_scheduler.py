import numpy as np


class TrainPipelineScheduler:
    def __init__(self, dataset_len, max_steps=None, max_epochs=None, eval_strategy="epochs", nsteps=None,
                 stop_patience=None, save_best=True):

        self.dataset_len = dataset_len
        self.eval_strategy = eval_strategy
        self.nsteps = nsteps
        self.stop_patience = stop_patience
        self.save_best = save_best

        self.steps_left = self._steps_left_from_max_steps_and_epochs(max_steps, max_epochs, dataset_len)
        self.best_model_save_id = None

    def has_to_save(self, eval_results):
        if not self.save_best:
            return False

        last_is_best = np.argmax(eval_results) == (len(eval_results) - 1)
        return last_is_best

    def decide_continue_training(self, eval_results):
        if self._check_steps_ok() and not self._check_stopping_patience_exceeded(eval_results):
            return True
        return False

    def get_train_kwargs(self):
        if self.eval_strategy == "epochs":
            train_kwargs = {"nepochs": 1}
        elif self.eval_strategy == "steps":
            if self.nsteps is None:
                raise ValueError("if eval_strategy is steps nsteps can't be None")
            if self.steps_left is None:
                nsteps = self.nsteps
            else:
                nsteps = min(self.nsteps, self.steps_left)
            train_kwargs = {"nsteps": nsteps}
        else:
            raise ValueError("incorrect eval_strategy_value")

        self._update_steps_left(train_kwargs)
        return train_kwargs

    def _check_steps_ok(self):
        if not self.steps_left is None:
            if self.steps_left <= 0:
                return False
        return True

    def _check_stopping_patience_exceeded(self, eval_results):
        if len(eval_results) != 0 and not self.stop_patience is None:
            steps_passed = len(eval_results)
            best_result_step = np.argmax(eval_results)
            steps_after_last_best_val = (steps_passed - best_result_step - 1)
            if steps_after_last_best_val >= self.stop_patience:
                print("EARLY STOPPING", "eval_vals", eval_results, "best_result_step", best_result_step)
                return True
        return False

    def _steps_left_from_max_steps_and_epochs(self, max_steps, max_epochs, steps_in_epoch):
        if not max_epochs is None:
            max_steps_because_of_epochs = max_epochs * steps_in_epoch
            if max_steps is None:
                max_steps = max_steps_because_of_epochs
            else:
                max_steps = min(max_steps, max_steps_because_of_epochs)
        return max_steps

    def _update_steps_left(self, train_kwargs):
        if not self.steps_left is None:
            if "nepochs" in train_kwargs:
                nepochs = train_kwargs["nepochs"]
                if not nepochs is None:
                    self.steps_left -= nepochs * self.dataset_len
            elif "nsteps" in train_kwargs:
                nsteps = train_kwargs["nsteps"]
                if not nsteps is None:
                    self.steps_left -= nsteps
