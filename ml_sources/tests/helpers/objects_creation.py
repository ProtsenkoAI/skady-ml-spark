import torch
from torch.utils import data as torch_data
import pandas as pd

from recsys_pipeline.data import datasets, loader_factories, datasets_retrievers
from recsys_pipeline.data_transform import preprocessing, id_idx_conv
from recsys_pipeline.managers import trainers, validators, train_eval_manager
from recsys_pipeline.models import mf_with_bias
from recsys_pipeline.saving import model_and_ids_saving, model_state_dict_saving
from . import tests_config
config = tests_config.TestsConfig()


def get_interacts(nrows=20):
    interacts = pd.read_csv(config.interacts_path, nrows=nrows*100)
    interacts = interacts[interacts["rating"] != -1] # clean up interacts without rating
    interacts = interacts.iloc[:nrows].reset_index(drop=True)

    users = interacts[config.user_colname]
    items = interacts[config.item_colname]
    user_id_idx_conv = get_id_converter()
    item_id_idx_conv = get_id_converter()
    interacts[config.user_colname] = user_id_idx_conv.add_ids_get_idxs(*users)
    interacts[config.item_colname] = item_id_idx_conv.add_ids_get_idxs(*items)

    return interacts


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
    dataloader = create_loader_from_dataset(dataset, batch_size)
    return dataloader


def get_datasets_retriever(interacts, batch_size=8):
    loader_builder = loader_factories.StandardLoaderBuilder(batch_size=batch_size)
    retriever = datasets_retrievers.UsersDatasetsRetriever(interacts, loader_builder)
    return retriever


def create_dataset_from_interacts(interacts):
    return datasets.InteractDataset(interacts)


def create_loader_from_dataset(dataset, batch_size=8):
    return torch_data.DataLoader(dataset, batch_size=batch_size)


def get_mf_model(nusers=20, nitems=20, **kwargs):
    return mf_with_bias.MFWithBiasModel(nusers, nitems, **kwargs)


def get_preprocessor():
    return preprocessing.TensorCreator(config.device)


def get_id_converter(*ids):
    return id_idx_conv.IdIdxConverter(*ids)


def get_trainer(model, interacts_nrows=20, batch_size=8):
    dataloader = get_loader(interacts_nrows, batch_size)
    preprocessor = get_preprocessor()
    trainer = trainers.Trainer(model, dataloader, preprocessor)
    return trainer


def get_validator(model, interact_nrows=20, batch_size=8):
    interacts = get_interacts(interact_nrows)
    retriever = get_datasets_retriever(interacts, batch_size)
    all_item_ids = get_all_item_ids(interact_nrows)
    preprocessor = get_preprocessor()

    validator = validators.Validator(model, retriever, all_item_ids,
                                     preprocessor)
    return validator


def get_meta_saver(default_models_dir="./"):
    return model_and_ids_saving.ModelAndIdsSaver(save_dir=default_models_dir)


def get_simple_saver(save_dir="./"):
    return model_state_dict_saving.ModelStateDictSaver(save_dir=save_dir)
