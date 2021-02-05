# TODO: add factories of complex objects to recsys_pipeline

from recsys_pipeline.saving import WeightsSaver
from recsys_pipeline.model_level.assistance import ModelAssistant
from recsys_pipeline.model_level.models import MFWithBiasModel
from recsys_pipeline.model_level.data_processing import DataProcessor, IdIdxConv
from recsys_pipeline.using_model_level.training import Trainer
from recsys_pipeline.using_model_level.evaluating import Validator

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


def get_trainer(assistant):
    return Trainer(assistant)


def get_validator(assistant):
    return Validator(assistant)
