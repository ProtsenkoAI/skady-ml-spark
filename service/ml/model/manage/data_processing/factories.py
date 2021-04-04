from .id_idx_conv import IdIdxConv
from .tensor_creation import TensorCreator
from .data_processor import DataProcessor


def get_standard_processor():
    user_conv = IdIdxConv()
    item_conv = IdIdxConv()
    tensorer = TensorCreator()
    processor = DataProcessor(user_conv, item_conv, tensor_creator=tensorer)
    return processor
