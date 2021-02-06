# TODO: add factories of complex objects to recsys_pipeline

from recsys_pipeline.saving import WeightsSaver, StandardSaver
from recsys_pipeline.model_level.assistance import ModelAssistant
from recsys_pipeline.model_level.models import MFWithBiasModel
from recsys_pipeline.model_level.data_processing import DataProcessor, IdIdxConv
from recsys_pipeline.using_model_level.training import Trainer
from recsys_pipeline.using_model_level.evaluating import Validator
from recsys_pipeline.main_tasks.train_pipeline_scheduler import TrainPipelineScheduler

from ..helpers import tests_config
config = tests_config.TestsConfig()


def get_assistant():
    model = MFWithBiasModel(1, 1, 1)
    user_conv = IdIdxConv(ids=[1, 2, 3])
    item_conv = IdIdxConv(ids=[1, 2, 3])
    data_processor = DataProcessor(user_conv, item_conv)
    return ModelAssistant(model, data_processor)


def get_weights_saver():
    return WeightsSaver(save_dir=config.save_dir)

def get_standard_saver():
    return StandardSaver(save_dir=config.save_dir)


def get_trainer():
    return Trainer()


def get_validator():
    return Validator()


def get_train_scheduler(steps_in_epoch, **kwargs):
    return TrainPipelineScheduler(steps_in_epoch, **kwargs)


def get_recommender():
    return None