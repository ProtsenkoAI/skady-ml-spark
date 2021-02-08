from ..helpers import std_objects
from recsys_pipeline.high_level_managing.train_pipeline import TrainPipelineManager


class ObjsPool:
    def __init__(self):
        # self.recommender = std_objects.get_recommender()
        self.recommender = None
        self.model = std_objects.get_model()
        self.assistant = std_objects.get_assistant()
        interacts = std_objects.get_interacts()
        self.assistant.update_with_interacts(interacts)

        self.trainer = std_objects.get_trainer()
        self.validator = std_objects.get_validator()
        self.train_dataloader = std_objects.get_dataloader()
        self.loader_builder = std_objects.get_dataloader_builder()

        self.standard_saver = std_objects.get_standard_saver()
        self.train_scheduler = std_objects.get_train_scheduler(len(self.train_dataloader), max_epochs=3)

        self.train_pipeline_manager = TrainPipelineManager(self.assistant, self.trainer, self.validator,
                                                           self.standard_saver, self.train_scheduler)
