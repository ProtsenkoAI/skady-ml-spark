class TrainPipelineManager:
    def __init__(self, model_manager, trainer, validator, saver, scheduler):
        self.model_manager = model_manager
        self.trainer = trainer
        self.validator = validator
        self.saver = saver
        self.scheduler = scheduler

        self.eval_results = []

    def run(self):
        while self.scheduler.decide_continue_training(self.eval_results):
            train_kwargs = self.scheduler.get_train_kwargs()
            self.trainer.fit(self.model_manager, **train_kwargs)
            eval_res = self.validator.evaluate(self.model_manager)
            self.eval_results.append(eval_res)

            if self.scheduler.has_to_save(self.eval_results):
                self.best_model_save_id = self.saver.save(self.model_manager)

        return self.eval_results

    def load_best(self):
        if self.best_model_save_id is None:
            raise ValueError("self.best_model_save_id is None. Probably you trying to load model before calling run()")
        return self.saver.load(self.best_model_save_id)
