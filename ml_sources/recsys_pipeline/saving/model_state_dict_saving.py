import os
import torch

from ..models import mf_with_bias


class ModelStateDictSaver:
    def __init__(self, save_dir="./", model_name="model_weights"):
        os.makedirs(save_dir, exist_ok=True)
        self.weights_path = os.path.join(save_dir, model_name + ".pt")

    def check_model_exists(self):
        if os.path.isfile(self.weights_path):
            return True
        return False

    def save(self, model):
        state = model.state_dict()
        torch.save(state, self.weights_path)

    def load(self, model):
        """Need model, because doesn't save any metadata. You can provide initialized, not trained model, and load()
        return model with saved parameters"""
        state = torch.load(self.weights_path)
        model.load_state_dict(state)
        return model
