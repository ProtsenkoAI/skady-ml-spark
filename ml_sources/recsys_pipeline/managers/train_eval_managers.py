import numpy as np


class TrainEvalManager:
    def __init__(self, trainer, validator, max_steps=None, max_epochs=None, 
                 eval_strategy="steps", nsteps=None, stopping_patience=None,
                 save_best_model=False, model_saver=None):
        """
        :param eval_strategy: str, one of (steps", "epoch", "patience")
        """
        self.trainer = trainer
        self.dataset_len = self.trainer.get_dataset_len()
        self.validator = validator
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.steps_passed = 0
        self.stopping_patience = stopping_patience
        self.eval_strategy = eval_strategy
        self.nsteps = nsteps
        self.save_best_model = save_best_model
        self.model_saver = model_saver
        self._validate_inp_args()

    def train_eval(self):
        self.eval_results = []
        while self._decide_train_continue():
            train_kwargs = self._get_train_kwargs_update_train_counters()
            self.trainer.fit(**train_kwargs)

            eval_res = self.validator.evaluate()
            self.eval_results.append(eval_res)

            if self._has_to_save_model():
                self.save_model()

        if self.save_best_model:
            best_model = self.load_best_model()
            return self.eval_results, best_model

        return self.eval_results

    def _get_train_kwargs_update_train_counters(self):
        """
        If training by epochs, return 1 epoch. If training by steps, check number 
        of steps left to max boundaries (max_steps and max_epochs) and adjust 
        number of steps for training if needed.
        """
        if self.eval_strategy == "epoch":
            train_kwargs = {"nepochs": 1}
        elif self.eval_strategy == "steps":
            steps = self._get_nsteps_adjusted_to_max_values()
            train_kwargs = {"nsteps": steps}
        else:
            raise ValueError("incorrect eval_strategy_value")
        self._update_train_counters()
        return train_kwargs

    def _get_nsteps_adjusted_to_max_values(self):
        if self.max_steps is None and self.max_epochs is None:
            return self.nsteps

        elif self.max_steps is None:
            max_epochs_steps = self.max_epochs * self.dataset_len
            nearest_maximum = max_epochs_steps
        elif self.max_epochs is None:
            nearest_maximum = self.max_steps
        else:
            max_epochs_steps = self.max_epochs * self.dataset_len
            max_steps_steps = self.max_steps
            nearest_maximum = min(max_epochs_steps, max_steps_steps)
        steps_left = nearest_maximum - self.steps_passed
        steps = min(self.nsteps, steps_left)
        return steps

    def _update_train_counters(self):
        if self.eval_strategy == "steps":
            self.steps_passed += self.nsteps
        else:
            self.steps_passed += self.dataset_len

    def _has_to_save_model(self):
        last_eval_is_best = np.argmax(self.eval_results) == (len(self.eval_results) - 1)
        if self.save_best_model and last_eval_is_best:
            return True
        return False

    def save_model(self):
        model = self.trainer.get_model()
        state_dict = model.state_dict()
        self.model_saver.save(state_dict)

    def load_best_model(self):
        # loading only state dict (parameters) - model changes only params during training
        model = self.trainer.get_model()
        state_dict = self.model_saver.load()
        model.load_state_dict(state_dict)
        return model

    def _decide_train_continue(self):
        if not self.max_steps is None:
            if self.max_steps <= self.trainer.get_steps_passed():
                return False
        if not self.max_epochs is None:
            if self.max_epochs <= self.trainer.get_epochs_passed():
                return False

        if not self.stopping_patience is None and len(self.eval_results): # refactor
            steps_passed = len(self.eval_results)
            best_result_step = np.argmax(self.eval_results)
            if (steps_passed - best_result_step - 1) >= self.stopping_patience:
                print("EARLY STOPPING", "eval_vals", self.eval_results, "best_result_step", best_result_step)
                return False
        return True

    def _validate_inp_args(self):
        if self.eval_strategy == "steps":
            if self.nsteps is None:
                raise ValueError("if eval_strategy == 'steps', then nsteps should be specified")
