class TrainPipelineManager:
    def __init__(self, model_manager, trainer, validator, saver, scheduler_builder):
        self.model_manager = model_manager
        self.trainer = trainer
        self.validator = validator
        self.saver = saver
        self.scheduler_builder = scheduler_builder

    def run(self, train_interacts, eval_dataset):
        # somehow pass number of steps in epoch to scheduler
        eval_results = []
        while scheduler.decide_continue_training(eval_results):
            train_kwargs = scheduler.get_train_kwargs()
            self.trainer.fit(self.model_manager, train_interacts, **train_kwargs)
            eval_res = self.validator.evaluate(self.model_manager, eval_dataset)
            eval_results.append(eval_res)

            if scheduler.has_to_save(eval_results):
                self.best_model_save_id = self.saver.save(self.model_manager)

        return eval_results

    def load_best(self):
        if self.best_model_save_id is None:
            raise ValueError("self.best_model_save_id is None. Probably you trying to load model before calling run()")
        return self.saver.load(self.best_model_save_id)
