import json
import os
import torch

from ..models import mf_with_bias
from ..data_transform import id_idx_conv


class ModelAndIdsSaver:
    def __init__(self, model_file_postfix="_weights", params_file_name="models_parameters.json",
                 save_dir="./", model_builder=mf_with_bias.MFWithBiasModel, model_name="model",
                 id_conv_creator=id_idx_conv.IdIdxConverter):

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_name = model_name

        self.model_builder = model_builder
        self.model_file_postfix = model_file_postfix
        self.params_path = os.path.join(save_dir, params_file_name)
        self.weights_path = self._weights_path_from_name(model_name)
        self.id_conv_creator = id_conv_creator

        self.model_data_template = {"conv_ids": {"users": None, "items": None},
                                    "model_init_kwargs": None,
                                    "model_weights_path": None}

    def check_model_exists(self):
        params = self._load_meta()
        if self.model_name in params.keys():
            return True
        return False

    def save(self, model, user_conv, item_conv):
        model_data = self._create_model_data(model, user_conv, item_conv)

        curr_models_dict = self._load_meta()
        curr_models_dict[self.model_name] = model_data
        self._save_model_and_data(model, curr_models_dict)

    def load(self):
        meta = self._load_meta()
        model_params = meta[self.model_name]
        return self._load_objects(model_params)

    def _create_model_data(self, model, user_conv, item_conv):

        model_data = self.model_data_template.copy()
        model_data["model_weights_path"] = self.weights_path
        model_data["model_init_kwargs"] = model.get_init_kwargs()
        model_data["conv_ids"]["users"] = user_conv.get_all_ids()
        model_data["conv_ids"]["items"] = item_conv.get_all_ids()

        return model_data

    def _save_model_and_data(self, model, data):
        torch.save(model.state_dict(), self.weights_path)
        with open(self.params_path, "w") as f:
            json.dump(data, f)

    def _load_objects(self, model_data):
        print("model_data", model_data)
        user_ids = model_data["conv_ids"]["users"]
        item_ids = model_data["conv_ids"]["items"]

        user_conv = self.id_conv_creator(*user_ids)
        item_conv = self.id_conv_creator(*item_ids)

        model = self.model_builder(**model_data["model_init_kwargs"])
        model_weights = torch.load(model_data["model_weights_path"])
        model.load_state_dict(model_weights)

        return model, (user_conv, item_conv)

    def _load_meta(self):
        if os.path.isfile(self.params_path):
            with open(self.params_path) as f:
                meta = json.load(f)
        else:
            meta = {}
        return meta

    def _weights_path_from_name(self, model_name):
        return os.path.join(self.save_dir, model_name + self.model_file_postfix + ".pt")
