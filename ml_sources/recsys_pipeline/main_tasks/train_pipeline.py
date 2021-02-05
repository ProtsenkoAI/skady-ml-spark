class TrainPipelineManager:
    def __init__(self, model_manager, trainer, validator, saver,
                 max_steps=None, max_epochs=None, eval_strategy="epochs"):
        self.model_manager = model_manager
        self.trainer = trainer
        self.validator = validator
        self.saver = saver

        

        self.best_model_save_id = None

    def run(self):
        self.best_model_save_id = self.saver.save(self.model_manager)

    def load_best(self):
        if self.best_model_save_id is None:
            raise ValueError("self.best_model_save_id is None. Probably you trying to load model before calling run()")
        return self.saver.load(self.best_model_save_id)
