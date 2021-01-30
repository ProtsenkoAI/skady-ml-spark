import json
import os
import torch
from ..models import mf_with_bias


class MetaModelSaver:
    def __init__(self, save_dir="./", params_file_name="models_parameters.json",
                 model_file_postfix="_weights", model_creator=mf_with_bias.MFWithBiasModel):
        self.save_dir = save_dir
        self.params_path = os.path.join(save_dir, params_file_name)
        self.model_file_postfix = model_file_postfix
        self.model_creator = model_creator

    def check_model_exists(self, model_name):
        params = self._load_params()
        if model_name in params.keys():
            return True
        return False

    def save(self, model_name, state_dict, model_init_kwargs):
        weights_path = self._weights_path_from_name(model_name)

        curr_dict = self._load_params()
        curr_dict[model_name] = model_init_kwargs
        torch.save(state_dict, weights_path)
        self._save_params(curr_dict)

    def load(self, model_name):
        weights_path = self._weights_path_from_name(model_name)

        params = self._load_params()
        model_params = params[model_name]

        model = self.model_creator(**model_params)
        model_weights = torch.load(weights_path)
        model.load_state_dict(model_weights)
        return model

    def _load_params(self):
        os.makedirs(self.save_dir, exist_ok=True) # create dir if needed
        if os.path.isfile(self.params_path):
            with open(self.params_path) as f:
                params = json.load(f)
        else:
            params = {}
        return params

    def _save_params(self, params):
        with open(self.params_path, "w") as f:
            json.dump(params, f)

    def _weights_path_from_name(self, model_name):
        return os.path.join(self.save_dir, model_name + self.model_file_postfix + ".pt")
