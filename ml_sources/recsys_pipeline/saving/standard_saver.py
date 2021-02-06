import os
import json
import torch

# TODO: delete all hardcodings
from ..model_level.data_processing import IdIdxConv
from ..model_level.models import MFWithBiasModel
from ..model_level.data_processing import DataProcessor
from ..model_level.assistance import ModelAssistant


class StandardSaver:
    def __init__(self, save_dir, weights_postfix="_weights", meta_postfix="_meta"):
        self.save_dir = save_dir
        self.weights_postfix = weights_postfix
        self.meta_postfix = meta_postfix
        os.makedirs(save_dir, exist_ok=True)

    def check_model_exists(self, model_name):
        weights_pth, _ = self._create_weights_and_meta_pathes(model_name)
        if os.path.isfile(weights_pth):
            return True
        return False

    def save(self, model_manager):
        model_name = "nigerundayo"
        model = model_manager.get_model()
        user_conv, item_conv = model_manager.get_convs()
        saved_user_conv_data = user_conv.dump()
        saved_item_conv_data = item_conv.dump()
        model_init_kwargs = model_manager.get_model_init_kwargs()

        weights_path, meta_path = self._create_weights_and_meta_pathes(model_name)

        meta = {"model_init_kwargs": model_init_kwargs, "user_conv": saved_user_conv_data,
                "item_conv": saved_item_conv_data}

        torch.save(model.state_dict(), weights_path)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        return model_name

    def load(self, model_name):
        weights_path, meta_path = self._create_weights_and_meta_pathes(model_name)

        with open(meta_path) as f:
            meta = json.load(f)

        state = torch.load(weights_path)

        model = MFWithBiasModel(**meta["model_init_kwargs"])
        model.load_state_dict(state)

        user_conv_data, item_conv_data = meta["user_conv"], meta["item_conv"]
        user_conv = IdIdxConv.load(user_conv_data)
        item_conv = IdIdxConv.load(user_conv_data)

        processor = DataProcessor(user_conv, item_conv)
        assistant = ModelAssistant(model, processor)
        return assistant

    def _create_weights_and_meta_pathes(self, model_name):
        weights_path = os.path.join(self.save_dir, model_name + self.weights_postfix + ".pt")
        meta_path = os.path.join(self.save_dir, model_name + self.meta_postfix + ".json")
        return weights_path, meta_path