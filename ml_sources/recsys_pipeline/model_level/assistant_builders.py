from .assistance import ModelAssistant
from .data_processing import DataProcessor, IdIdxConv, TensorCreator


class AssistantBuilder:
    def __init__(self, base_model_class, **model_kwargs):
        self.base_model_class = base_model_class
        self.model_kwargs = model_kwargs

    def build(self):
        model = self.base_model_class(**self.model_kwargs)
        user_conv = IdIdxConv()
        item_conv = IdIdxConv()
        tensor_creator = TensorCreator()
        processor = DataProcessor(user_conv, item_conv, tensor_creator)
        assistant = ModelAssistant(model, processor)
        return assistant
