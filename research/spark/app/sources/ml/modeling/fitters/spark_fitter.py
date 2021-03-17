from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StringType, StructField
from collections import namedtuple
import os
import codecs
import dill
from torch import nn, optim

# TODO: delete reading config here and move it to top level
from util import read_config

conf = read_config()
SparkRes = DataFrame  # TODO: move to types

SerializablePackagedModel = namedtuple("SerializablePackagedModel", ["model", "optimizer_class", "criterion", "optim_params"])
ModelObj = SerializablePackagedModel
FitBatchRes = DataFrame


class SparkFitter:
    # TODO: check that serialises well
    def __init__(self, params: dict, model, lr=1e-4):
        self.params = params
        self.batch_size = params["batch_size"]
        self.nsteps = params["nsteps"]
        self.fit_out_schema = StructType([
            StructField("batch_fit_out", StringType())
        ])
        self.path2packed_model = os.path.join(conf["base_path"], conf["worker_dir"], "torch_obj.dill")

        model = model
        criterion = nn.MSELoss()
        optimizer = optim.SparseAdam
        opt_params = {"lr": lr}
        fit_obj = self._create_fit_obj(model, criterion, optimizer, opt_params)
        self._save_fit_obj(fit_obj)

    def fit_model(self, data: DataFrame) -> SparkRes:
        data.writeStream.
        # fit_res = data.foreachPartition(lambda batch: self._fit_batch_save_model(batch))
        # return fit_res

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
        os.makedirs(os.path.dirname(self.path2packed_model), exist_ok=True)
        with open(self.path2packed_model, "wb") as f:
            obj = codecs.encode(dill.dumps(fit_obj), "base64")
            f.write(dill.dumps(obj))

    def _load_fit_obj_from_path(self, path):
        with codecs.open(path, "rb") as f:
            torch_obj_decoded = codecs.decode(f, "base64")
        torch_obj = dill.load(torch_obj_decoded)
        return torch_obj

    def _fit_batch_save_model(self, batch) -> FitBatchRes:
        # TODO: check that all used objects are correctly serializable
        print("batch", batch)
        fit_obj = self._load_fit_obj()
        raise NotImplementedError
