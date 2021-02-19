import pandas as pd
from torch.utils import data as torch_data

from saving.savers import StandardSaver
from storages import LocalModelStorage
from model_level.assistance import ModelAssistant
from model_level.models import MFWithBiasModel
from model_level.data_processing import get_standard_processor
from data.building_loaders import StandardLoaderBuilder, UserItemsLoaderBuilder
from data.datasets import InteractDataset
from model_level.recommender import Recommender
from train_eval.training import SimpleTrainer
from train_eval.evaluation import Validator
from train_eval.training.trainer_with_eval import EvalTrainer

from helpers import tests_config

config = tests_config.TestsConfig()


def get_model(nusers=5, nitems=5, hidden_size=5):
    model = MFWithBiasModel(nusers, nitems, hidden_size)
    return model


def get_assistant(nusers=5, nitems=5, hidden=5):
    model = MFWithBiasModel(nusers, nitems, hidden)
    return ModelAssistant(model, get_standard_processor())


def get_standard_saver():
    save_storage = LocalModelStorage(save_dir=config.save_dir)
    return StandardSaver(save_storage)


def get_simple_trainer():
    loader_builder = get_dataloader_builder(batch_size=6)
    return SimpleTrainer(loader_builder)


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


def get_eval_trainer(**eval_trainer_kwargs):
    return EvalTrainer(get_validator(), get_dataloader_builder(),
                       get_standard_saver(), **eval_trainer_kwargs)
