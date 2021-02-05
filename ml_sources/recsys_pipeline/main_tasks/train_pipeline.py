from . import train_pipeline_helpers as helpers


class TrainPipelineManager:
    def __init__(self, model_manager, trainer, validator, saver,
                 max_steps=None, max_epochs=None, eval_strategy="epochs", nsteps=None,
                 stop_patience=None, save_best=True):
        self.model_manager = model_manager
        self.trainer = trainer
        self.validator = validator
        self.saver = saver

        self.steps_left = helpers.steps_left_from_max_steps_epochs(max_steps, max_epochs, self.trainer.get_dataset_len())
        self.eval_strategy = eval_strategy
        self.nsteps = nsteps
        self.stop_patience = stop_patience
        self.save_best = save_best

        self.best_model_save_id = None
        self.eval_results = []

    def run(self):
        while helpers.decide_continue_training(self.eval_results, self.stop_patience, self.steps_left):
            train_kwargs = helpers.get_train_kwargs(self.nsteps, self.steps_left, self.eval_strategy)
            self.trainer.fit(**train_kwargs)
            eval_res = self.validator.evaluate()
            self.eval_results.append(eval_res)

            if helpers.has_to_save(self.eval_results, self.save_best):
                self.best_model_save_id = self.saver.save(self.model_manager)

            self.steps_left = helpers.update_steps_left(self.steps_left, train_kwargs, self.trainer.get_dataset_len())

        return self.eval_results

    def load_best(self):
        if self.best_model_save_id is None:
            raise ValueError("self.best_model_save_id is None. Probably you trying to load model before calling run()")
        return self.saver.load(self.best_model_save_id)
