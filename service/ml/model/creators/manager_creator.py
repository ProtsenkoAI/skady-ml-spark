from model.expose.model_manager import ModelManager
# TODO: add processor creator later (now the line below violates layering)
from model.manage.data_processing import SparkProcessor, IdIdxConv, TensorCreator
from model.manage.models import MFWithBiasModel
import os


class ManagerCreator:
    def __init__(self, config, common_params):
        params = config["model_manager_params"]
        paths_info = common_params["paths"]
        self.device = params["device"]
        self.nusers = params["nusers"]
        self.nitems = self.nusers
        self.hidden_size = params["hidden_size"]
        self.mode = "spark"  # TODO: delete when will refactor obtainer so processor can be agnostic to source

        self.model_path = os.path.join(paths_info["base_path"],
                                       paths_info["worker_dir"],
                                       paths_info["model_checkpoint_name"])

        if self.mode not in ["spark"]:
            raise NotImplementedError

    def get(self):
        # TODO: add ability to load manager
        # TODO: add force_create flag for model
        user_conv = IdIdxConv()
        item_conv = IdIdxConv()
        tensor_creator = TensorCreator(device=self.device)

        # TODO: instead of hardcoded model, can pass model_params to model fabric.
        torch_model = MFWithBiasModel(self.nusers, self.nitems, self.hidden_size)

        processor = SparkProcessor(user_conv, item_conv, tensor_creator)
        model_manager = ModelManager(torch_model, processor)
        return model_manager
