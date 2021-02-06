import os
import json


class StandardSaver:
    def __init__(self, save_dir, meta_filename="meta"):
        self.save_dir = save_dir
        self.meta_path = os.path.join(self.save_dir, meta_filename + "json")
        os.makedirs(save_dir, exist_ok=True)

    def check_model_exists(self, model_name):
        meta = self._load_meta()

    def _load_meta(self):