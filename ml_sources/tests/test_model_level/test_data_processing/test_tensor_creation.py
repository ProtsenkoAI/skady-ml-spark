import unittest
import torch

from model_level.data_processing.tensor_creation import TensorCreator
from ...helpers import std_objects, tests_config
config = tests_config.TestsConfig()


class TestTensorCreation(unittest.TestCase):
    def setUp(self):
        self.tensor_creator = TensorCreator()

    def test_get_feature_tensor(self):
        users = [[1], [1]]
        items = [[2], [3]]
        users_tensor = self.tensor_creator.get_feature_tensor(users)
        items_tensor = self.tensor_creator.get_feature_tensor(items)

        for tensor in [users_tensor, items_tensor]:
            self.assertEqual(len(tensor), 2)
            self.assertIsInstance(tensor, torch.Tensor)

    def test_get_label_tensor(self):
        labels = [[-1], [6]]
        tensor = self.tensor_creator.get_labels_tensor(labels)
        self.assertEqual(len(tensor), len(labels))
        self.assertIsInstance(tensor, torch.Tensor)
