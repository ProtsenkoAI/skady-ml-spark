from model.expose.model_manager import ModelManager
# TODO: add processor creator later (now the line below violates layering)
from model.manage.data_processing import SparkProcessor, IdIdxConv, TensorCreator
from model.manage.models import MFWithBiasModel, MFWithUserEmbeddings

from .saver_creator import SaverCreator
import os


class ManagerCreator:
    def __init__(self, config, common_params):
        self.paths = config["common_params"]["paths"]
        params = config["model_manager_params"]
        self.params = params
        self.device = params["device"]
        self.nusers = params["nusers"]
        self.nitems = self.nusers
        self.hidden_size = params["hidden_size"]
        self.mode = "spark"  # TODO: delete when will refactor obtainer so processor can be agnostic to source

        self.manager_saver = SaverCreator(config, common_params).get_manager_saver()

        if self.mode not in ["spark"]:
            raise NotImplementedError

    def get(self):
        # TODO: add ability to load manager
        # TODO: add force_create flag for model
        user_conv = IdIdxConv()
        item_conv = IdIdxConv()
        tensor_creator = TensorCreator(device=self.device)

        # TODO: instead of hardcoded model, can pass model_params to model fabric.

        if self.params["use_vk"]:
            embeddings_path = os.path.join(self.paths["base_path"],
                                           self.paths["worker_dir"],
                                           self.paths["embeddings_file"])
            torch_model = MFWithUserEmbeddings(self.nusers, self.nitems, self.hidden_size, embeddings_path, 300)

        else:
            torch_model = MFWithBiasModel(self.nusers, self.nitems, self.hidden_size)

        processor = SparkProcessor(user_conv, item_conv, tensor_creator)
        model_manager = ModelManager(torch_model, processor, self.manager_saver)
        return model_manager
