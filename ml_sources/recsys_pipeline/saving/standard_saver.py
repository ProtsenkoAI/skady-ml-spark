import os
import json
import torch
import uuid

from ..model_level.data_processing import IdIdxConv, TensorCreator
from ..model_level.models import MFWithBiasModel
from ..model_level.data_processing import DataProcessor
from ..model_level.assistance import ModelAssistant


class StandardSaver:
    def __init__(self, save_dir, weights_postfix="_weights", meta_file_name="meta_info"):
        self.save_dir = save_dir
        self.weights_postfix = weights_postfix
        self.meta_path = os.path.join(self.save_dir, meta_file_name + ".json")
        self.tensor_creator = TensorCreator()
        os.makedirs(save_dir, exist_ok=True)

    def check_model_exists(self, model_name):
        weights_pth = self._get_weights_path(model_name)
        if os.path.isfile(weights_pth):
            return True
        return False

    def save(self, model_manager):
        meta = self._load_meta()
        model_name = self._create_model_name(meta)
        new_meta = self.update_meta(meta, model_name, model_manager)

        model = model_manager.get_model()
        self._save_model_and_meta(model, new_meta, model_name)

        return model_name

    def update_meta(self, meta, model_name, model_manager):
        user_conv, item_conv = model_manager.get_convs()
        saved_user_conv_data = user_conv.dump()
        saved_item_conv_data = item_conv.dump()
        model_init_kwargs = model_manager.get_model_init_kwargs()
        meta[model_name] = {"model_init_kwargs": model_init_kwargs, "user_conv": saved_user_conv_data,
                            "item_conv": saved_item_conv_data}
        return meta

    def load(self, model_name):
        weights_path = self._get_weights_path(model_name)

        meta = self._load_meta()

        state = torch.load(weights_path)

        model = MFWithBiasModel(**meta[model_name]["model_init_kwargs"])
        model.load_state_dict(state)

        user_conv_data, item_conv_data = meta[model_name]["user_conv"], meta[model_name]["item_conv"]
        user_conv = IdIdxConv.load(user_conv_data)
        item_conv = IdIdxConv.load(item_conv_data)

        processor = DataProcessor(user_conv, item_conv, self.tensor_creator)
        assistant = ModelAssistant(model, processor)
        return assistant

    def _create_model_name(self, meta):
        existing_models = meta.keys()
        model_name = str(uuid.uuid4())
        while model_name in existing_models:
            model_name = str(uuid.uuid4())
        return model_name

    def _load_meta(self):
        if os.path.isfile(self.meta_path):
            with open(self.meta_path) as f:
                return json.load(f)
        else:
            return {}

    def _save_model_and_meta(self, model, meta, model_name):
        weights_path = self._get_weights_path(model_name)
        torch.save(model.state_dict(), weights_path)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

    def _get_weights_path(self, model_name):
        weights_path = os.path.join(self.save_dir, model_name + self.weights_postfix + ".pt")
        return weights_path
