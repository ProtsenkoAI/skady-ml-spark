from ..helpers import std_objects
from recsys_pipeline.main_tasks.train_pipeline import TrainPipelineManager


class ObjsPool:
    def __init__(self):
        self.recommender = std_objects.get_recommender()
        self.assistant = std_objects.get_assistant()
        self.trainer = std_objects.get_trainer()
        self.validator = std_objects.get_validator()
        self.weights_saver = std_objects.get_weights_saver()
        self.standard_saver = std_objects.get_standard_saver()
        self.train_scheduler = std_objects.get_train_scheduler(self.trainer.get_dataset_len(), max_epochs=3)

        self.train_pipeline_manager = TrainPipelineManager(self.assistant, self.trainer, self.validator,
                                                           self.weights_saver, self.train_scheduler)
