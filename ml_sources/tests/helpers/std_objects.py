# TODO: add factories of complex objects to recsys_pipeline
import pandas as pd
from torch.utils import data as torch_data

from recsys_pipeline.saving import StandardSaver
from recsys_pipeline.model_level.assistance import ModelAssistant
from recsys_pipeline.model_level.models import MFWithBiasModel
from recsys_pipeline.model_level.data_processing import DataProcessor, IdIdxConv, TensorCreator
from recsys_pipeline.data.building_loaders import StandardLoaderBuilder, UserItemsLoaderBuilder
from recsys_pipeline.data.datasets import InteractDataset
from recsys_pipeline.model_level.recommender import Recommender
from recsys_pipeline.trains_evals.training import Trainer
from recsys_pipeline.trains_evals.evaluation import Validator
from recsys_pipeline.high_level_managing.train_pipeline_scheduler import TrainPipelineScheduler

from ..helpers import tests_config
config = tests_config.TestsConfig()


def get_model(nusers=5, nitems=5, hidden_size=5):
    model = MFWithBiasModel(nusers, nitems, hidden_size)
    return model


def get_processor():
    user_conv = IdIdxConv()
    item_conv = IdIdxConv()
    tensorer = TensorCreator()
    processor = DataProcessor(user_conv, item_conv, tensor_creator=tensorer)
    return processor


def get_assistant(nusers=5, nitems=5, hidden=5):
    model = MFWithBiasModel(nusers, nitems, hidden)
    return ModelAssistant(model, get_processor())


def get_standard_saver():
    return StandardSaver(save_dir=config.save_dir)


def get_trainer():
    return Trainer()


def get_validator():
    recommender = get_recommender()
    return Validator(recommender)


def get_train_scheduler(steps_in_epoch, **kwargs):
    return TrainPipelineScheduler(steps_in_epoch, **kwargs)


def get_recommender():
    builder = UserItemsLoaderBuilder(batch_size=8)
    # return Recommender(builder)
    return Recommender(builder)


def get_interacts(nrows=20):
    return pd.read_csv(config.interacts_path, nrows=nrows)


def get_dataloader(batch_size=8, interacts=None):
    if interacts is None:
        interacts = get_interacts()
    dataset = InteractDataset(interacts)
    return torch_data.DataLoader(dataset, batch_size=batch_size)


def get_dataloader_builder(batch_size=8):
    return StandardLoaderBuilder(batch_size)
