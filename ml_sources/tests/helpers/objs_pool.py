from ..helpers import std_objects
from recsys_pipeline.main_tasks.train_pipeline import TrainPipelineManager


class ObjsPool:
    def __init__(self):
        self.assistant = std_objects.get_assistant()
        self.trainer = std_objects.get_trainer(self.assistant)
        self.validator = std_objects.get_validator(self.assistant)
        self.weights_saver = std_objects.get_weights_saver()

        self.train_pipeline_manager = TrainPipelineManager(self.assistant, self.trainer, self.validator,
                                                           self.weights_saver, max_epochs=3)
