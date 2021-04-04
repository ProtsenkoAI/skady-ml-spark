from model.manage.data_processing import IdIdxConv, TensorCreator
from model.manage.models import MFWithBiasModel
from model.manage.data_processing import DataProcessor


class StandardSaver:
    # TODO: refactor and set up interfaces
    def __init__(self, model_storage):
        self.model_storage = model_storage
        self.tensor_creator = TensorCreator()

    def check_model_exists(self, model_name):
        return self.model_storage.check_model_exists(model_name)

    def save(self, model, processor):
        model_meta = self._get_model_meta(model, processor)
        model_weights = model.state_dict()
        model_name = self.model_storage.save_weights_and_meta(model_weights, model_meta)

        return model_name

    def _get_model_meta(self, model, processor):
        user_conv_data, item_conv_data = processor.get_convs_data()
        model_init_kwargs = model.get_init_kwargs()
        model_meta = {"model_init_kwargs": model_init_kwargs, "user_conv": user_conv_data,
                            "item_conv": item_conv_data}
        return model_meta

    def load(self, model_name):
        model_weights, model_meta = self.model_storage.load_weights_and_meta(model_name)
        model = MFWithBiasModel(**model_meta["model_init_kwargs"])
        model.load_state_dict(model_weights)

        user_conv_data, item_conv_data = model_meta["user_conv"], model_meta["item_conv"]
        user_conv = IdIdxConv.load(user_conv_data)
        item_conv = IdIdxConv.load(item_conv_data)

        processor = DataProcessor(user_conv, item_conv, self.tensor_creator)
        return model, processor
