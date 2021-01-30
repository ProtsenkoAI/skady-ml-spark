import torch
from torch.utils import data as torch_data
import pandas as pd


from recsys_pipeline.models import mf_with_bias
from recsys_pipeline.data import datasets
from recsys_pipeline.data_transform import data_preprocessor, id_idx_converter
from ..helpers import tests_config

config = tests_config.TestsConfig()

def get_model(nusers, nitems):
    return mf_with_bias.MFWithBiasModel(nusers, nitems)

def load_interacts(interacts_nrows):
    interacts = pd.read_csv(config.interacts_path, nrows=interacts_nrows)

    user_id_idx_conv = id_idx_converter.IdIdxConverter(*interacts["user_id"].unique())
    interacts["user_id"] = user_id_idx_conv.get_idxs(*interacts["user_id"])

    item_id_idx_conv = id_idx_converter.IdIdxConverter(*interacts["anime_id"].unique())
    interacts["anime_id"] = item_id_idx_conv.get_idxs(*interacts["anime_id"])
    return interacts

def create_dataloader(interacts):

    dataset = datasets.InteractDataset(interacts)
    dataloader = torch_data.DataLoader(dataset, batch_size=8)
    return dataloader

def get_preprocessor():
    return data_preprocessor.DataPreprocessor(DEVICE)