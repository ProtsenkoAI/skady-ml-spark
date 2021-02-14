class TrainPipelineManager:
    def __init__(self, model_manager, trainer, validator, saver, scheduler_builder, loaders_generator_creator):
        self.model_manager = model_manager
        self.trainer = trainer
        self.validator = validator
        self.saver = saver
        self.scheduler_builder = scheduler_builder
        self.loaders_generator_creator = loaders_generator_creator

    def run(self, train_interacts, eval_dataset, steps_per_iter=None, epochs_per_iter=None):
        # somehow pass number of steps in epoch to scheduler
        eval_results = []
        loaders_generator = self.loaders_generator_creator.build(train_interacts, steps_per_iter, epochs_per_iter)
        steps_in_epoch = loaders_generator.get_nsteps_in_epochs(train_interacts)
        scheduler = self.scheduler_builder(steps_in_epoch)

        # while scheduler.decide_continue_training(eval_results):
        for loader in loaders_generator:
            if not scheduler.decide_continue_training(eval_results):
                break

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
