import os
import torch


class ModelStateDictSaver:
    def __init__(self, save_dir="./", model_file_name="model_weights"):
        self.save_dir = save_dir
        self.state_file_name = os.path.join(save_dir, model_file_name + ".pt")

    def save(self, state_dict):
        torch.save(state_dict, self.state_file_name)

    def load(self):
        return torch.load(self.state_file_name)
