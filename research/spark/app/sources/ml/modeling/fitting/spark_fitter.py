# TODO: test time to load and save model
# TODO: maybe learn from tests structure of sparttorch project on github
# TODO: finish type hints

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StringType, StructField
from collections import namedtuple
import os
import codecs
import dill
from torch import nn, optim
from torch.utils import data as torch_data
from time import time

# TODO: delete reading config here and move it to top level
from util import read_config
from .pytorch_fitting.weights_updating import TorchWeightsUpdater


conf = read_config()
# TODO: refactor types
SparkRes = DataFrame  # TODO: move to types

SerializablePackagedModel = namedtuple("SerializablePackagedModel",
                                       ["model", "optimizer_class", "criterion", "optim_params"])

# optim params are needed to save optim later
# ModelObj = namedtuple("ModelObj", ["model", "optimizer", "criterion", "optim_params"])
FitBatchRes = DataFrame


class ModelObj:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion, optim_params=None,
                 lr=1e-4):
        if optim_params is None:
            optim_params = {}

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.optim_params = optim_params
        self.weights_updater = TorchWeightsUpdater(optimizer, criterion)

    def fit(self, data: torch_data.DataLoader):
        print("new_fit")
        for batch in data:
            self.weights_updater.fit_with_batch(self.model, batch)

    def convert_to_serializable(self) -> SerializablePackagedModel:
        optimizer_class = type(self.optimizer)
        fit_obj = SerializablePackagedModel(model=self.model,
                                            optimizer_class=optimizer_class,
                                            optim_params=self.optim_params,
                                            criterion=self.criterion)
        return fit_obj


class SparkFitter:
    # TODO: check that serialises well
    def __init__(self, params: dict, model, preprocessor, lr=1e-4):
        self.params = params
        self.batch_size = params["batch_size"]
        self.nsteps = params["nsteps"]
        self.fit_out_schema = StructType([
            StructField("batch_fit_out", StringType())
        ])
        self.path2packed_model = os.path.join(conf["base_path"], conf["worker_dir"], "torch_obj.dill")

        model = model
        self.preprocessor = preprocessor
        criterion = nn.MSELoss()
        optimizer = optim.SparseAdam
        opt_params = {"lr": lr}
        fit_obj = self._create_fit_obj(model, criterion, optimizer, opt_params)
        self._save_fit_obj(fit_obj)

    def fit_model(self, data: DataFrame):
        def fit_on_batch(batch, batch_idx):
            self._fit_batch_save_model(batch)

        fit_queue = data.writeStream.outputMode("append").foreachBatch(fit_on_batch).start()
        fit_queue.awaitTermination()

    def _load_fit_obj(self) -> ModelObj:
        if os.path.isfile(self.path2packed_model):
            return self._load_fit_obj_from_path(self.path2packed_model)
        else:
            raise ValueError(f"path2packed_model: {self.path2packed_model} doesn't exist")

    def _create_fit_obj(self, model, criterion, optimizer, opt_params) -> SerializablePackagedModel:
        return SerializablePackagedModel(
                model=model,
                criterion=criterion,
                optimizer_class=optimizer,
                optim_params=opt_params
        )

    def _save_fit_obj(self, fit_obj: SerializablePackagedModel):
        start = time()
        os.makedirs(os.path.dirname(self.path2packed_model), exist_ok=True)
        serialized = dill.dumps(fit_obj)
        with open(self.path2packed_model, "wb") as f:
            encoded = codecs.encode(serialized, "base64")
            f.write(encoded)
        print("time to save", time() - start)

    def _load_fit_obj_from_path(self, path):
        start = time()
        with codecs.open(path, "rb") as f:
            torch_obj_decoded = codecs.decode(f.read(), "base64")
        loaded_obj = dill.loads(torch_obj_decoded)
        model, criterion = loaded_obj.model, loaded_obj.criterion
        optimizer = loaded_obj.optimizer_class(model.parameters(), **loaded_obj.optim_params)
        print("time to load", time() - start)
        return ModelObj(model, optimizer, criterion, loaded_obj.optim_params)

    def _fit_batch_save_model(self, batch_raw):
        # TODO: check that all used objects are correctly serializable
        # TODO: maybe create separate object with everything needed for training, loading and saving
        fit_obj = self._load_fit_obj()
        batch = self.preprocessor.preproc(batch_raw)
        fit_obj.fit(batch)
        serializable_fit_obj = fit_obj.convert_to_serializable()
        self._save_fit_obj(serializable_fit_obj)
