from .data_managers.fit_data_managers import FitDataManager
from pyspark.sql import DataFrame
from torch.utils import data as torch_data
import pandas as pd
import numpy as np

FitData = DataFrame
TorchFitInp = torch_data.DataLoader


class FitDataPreprocessor:
    def __init__(self, data_manager: FitDataManager, mode="torch"):
        """
        :param data_manager: technology-specific (spark, local-machine fit, distributed etc) object
            with interface to process data
        """
        # TODO: not to hardcode label_colname
        self.data_manager = data_manager
        if mode == "torch":
            self.fit_inp_builder = TorchInpBuilder(label_colname="label")
        else:
            raise ValueError(mode)

    def create_fit_data(self, new_data_raw: DataFrame) -> FitData:
        fit_data = self.data_manager.prepare_for_fit(new_data_raw)
        return fit_data

    def preproc(self, data: DataFrame) -> TorchFitInp:
        pd_df = data.toPandas()
        return self.fit_inp_builder(pd_df)


class TorchInpBuilder:
    # TODO: it's fast-built realization, we have good realization at recsys_pipeline/data
    def __init__(self, label_colname, batch_size=8):
        self.label_colname = label_colname
        self.batch_size = batch_size

    def __call__(self, pdf: pd.DataFrame) -> torch_data.DataLoader:
        labels = pdf[self.label_colname]
        other_cols = [col for col in pdf.columns if col != self.label_colname]
        features = pdf[other_cols]
        dataset = SimpleDataset(features, labels)
        return torch_data.DataLoader(dataset, batch_size=self.batch_size)


class SimpleDataset(torch_data.Dataset):
    def __init__(self, *dataframes):
        """
        :indexed_objs: have to assert __getitem__ and len
        """
        self.dfs = dataframes
        lenghts = [len(obj) for obj in self.dfs]
        all_len_equal = lenghts.count(lenghts[0]) == len(lenghts)
        assert all_len_equal
        self.len = lenghts[0]

    def __getitem__(self, idx):
        out = []
        for df in self.dfs:
            out.append(np.array(df.iloc[idx]))
        return out

    def __len__(self):
        return self.len
