from .model_managing import ModelManager
from .data_processing import DataProcessor, SparkProcessor, IdIdxConv, TensorCreator
from .models import MFWithBiasModel


class ManagerCreator:
    def __init__(self, params):
        self.device = params["device"]
        self.nusers = params["nusers"]
        self.nitems = self.nusers
        self.hidden_size = params["hidden_size"]
        self.mode = "spark"  # TODO: delete when will refactor obtainer so processor can be agnostic to source

    def get(self):
        # TODO: add ability to load manager
        # TODO: add force_create flag for model
        user_conv = IdIdxConv()
        item_conv = IdIdxConv()
        tensor_creator = TensorCreator(device=self.device)
        if self.mode == "local":
            processor = DataProcessor(user_conv, item_conv, tensor_creator)
        elif self.mode == "spark":
            processor = SparkProcessor(user_conv, item_conv, tensor_creator)

        # TODO: instead of hardcoded model, can pass model_params to model fabric.
        torch_model = MFWithBiasModel(self.nusers, self.nitems, self.hidden_size)
        model_manager = ModelManager(torch_model, processor)
        return model_manager
