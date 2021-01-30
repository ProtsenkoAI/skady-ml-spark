import torch
import pandas as pd

from recsys_pipeline.data import *
from recsys_pipeline.data_transform import *
from recsys_pipeline.managers import *
from recsys_pipeline.models import mf_with_bias
from recsys_pipeline.saving import *
from . import tests_config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_interacts(nrows=20):
    return pd.read_csv(config.interacts_path, nrows=nrows)

def get_all_item_ids(nrows=20):
    interacts = get_interacts(nrows)
    all_item_ids = interacts["anime_id"].unique()
    return all_item_ids


def get_interact_dataset(nrows=20):
    interacts = get_interacts(nrows)
    dataset = datasets.InteractDataset(interacts)
    return dataset


def get_loader(nrows=20, batch_size=8):
    dataset = get_interact_dataset(nrows)
    dataloader = torch_data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


def get_mf_model(nusers=20, nitems=20):
    return mf_with_bias.MFWithBiasModel(nusers, nitems)


def get_preprocessor():
    return data_preprocessor.DataPreprocessor(DEVICE)


def get_id_converter(*ids):
    return id_idx_converter.IdIdxConverter(*ids)


def get_trainer(model, interacts_nrows=20, batch_size=8):
    dataloader = get_loader(interacts_nrows, batch_size)
    preprocessor = get_preprocessor()
    trainer = trainers.Trainer(model, dataloader, preprocessor)
    return trainer


def get_validator(model, interact_nrows=20, batch_size=8):
    dataloader = get_loader(interact_nrows, batch_size)
    all_item_ids = get_all_item_ids(interact_nrows)
    preprocessor = get_preprocessor()

    validator = validators.Validator(model, dataset, all_item_ids, 
                                     preprocessor)
    return validator


def get_meta_saver(default_models_dir="./"):
    return meta_model_saving.MetaModelSaver(save_dir=default_models_dir)


def get_simple_saver(save_dir="./"):
    return model_state_dict_saving.ModelStateDictSaver(save_dir=save_dir)
