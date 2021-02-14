import os
import uuid
import json
import torch


class LocalModelStorage:
    def __init__(self, save_dir, weights_postfix="_weights", meta_file_name="meta_info"):
        self.save_dir = save_dir
        self.weights_postfix = weights_postfix
        self.meta_path = os.path.join(self.save_dir, meta_file_name + ".json")
        os.makedirs(save_dir, exist_ok=True)

    def check_model_exists(self, model_name):
        weights_path = self._get_weights_path(model_name)
        return os.path.isfile(weights_path)

    def delete_model(self, model_name):
        meta = self._load_meta()
        del meta[model_name]
        weights_path = self._get_weights_path(model_name)
        os.remove(weights_path)

    def save_weights_and_meta(self, model_weights: dict, model_meta: dict):
        all_model_meta = self._load_meta()
        model_name = self._create_model_name(all_model_meta)
        all_model_meta[model_name] = model_meta
        weights_path = self._get_weights_path(model_name)

        torch.save(model_weights, weights_path)
        self._save_meta(all_model_meta)

        return model_name

    def load_weights_and_meta(self, model_name):
        weights_path = self._get_weights_path(model_name)
        model_meta = self._load_meta()[model_name]
        state = torch.load(weights_path)
        return state, model_meta

    def _create_model_name(self, meta):
        existing_models = meta.keys()
        model_name = str(uuid.uuid4())
        while model_name in existing_models:
            model_name = str(uuid.uuid4())
        return model_name

    def _save_meta(self, meta):
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

    def _load_meta(self):
        if os.path.isfile(self.meta_path):
            with open(self.meta_path) as f:
                return json.load(f)
        else:
            return {}

    def _get_weights_path(self, model_name):
        weights_path = os.path.join(self.save_dir, model_name + self.weights_postfix + ".pt")
        return weights_path
