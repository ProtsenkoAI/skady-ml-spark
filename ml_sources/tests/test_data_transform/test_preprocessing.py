import unittest
import numpy as np
import torch

from recsys_pipeline.data_transform import preprocessing
from ..helpers import tests_config
config = tests_config.TestsConfig()


class TestDataPreprocessor(unittest.TestCase):
    """The main point is: whatever data are, outputs must be tensors"""
    def setUp(self):
        self.batchsize = 8
        self.preprocessor = preprocessing.DataPreprocessor(config.device)

        # creating standard input data
        self.users = np.full(self.batchsize, 0)
        self.items = np.full(self.batchsize, 1)
        self.labels = np.full(self.batchsize, 2)
        self.features = np.concatenate([self.users.reshape(-1, 1), self.items.reshape(-1, 1)], axis=1)
        self.batch = [self.features, self.labels]

    def test_transform_batch_types(self):
        out = self.preprocessor.preprocess_batch(self.batch)
        (proc_users, proc_items), proc_labels = out
        
        for out_tensor in [proc_users, proc_items, proc_labels]:
            self.assertIsInstance(out_tensor, torch.Tensor)

    def test_transform_batch_shapes(self):
        out = self.preprocessor.preprocess_batch(self.batch)
        (proc_users, proc_items), proc_labels = out

        before_after_pairs = [(self.users, proc_users),
                              (self.items, proc_items),
                              (self.labels, proc_labels)]

        for before, after in before_after_pairs:
            len1 = len(before)
            len2 = after.shape[0]
            self.assertEqual(len1, len2)

    def test_transform_users(self):
        proc_users = self.preprocessor.preprocess_users(self.users)
        len1 = len(self.users)
        len2 = proc_users.shape[0]

        self.assertIsInstance(proc_users, torch.Tensor)
        self.assertEqual(len1, len2)

    def test_transform_items(self):
        proc_items = self.preprocessor.preprocess_items(self.items)
        len1 = len(self.items)
        len2 = proc_items.shape[0]

        self.assertIsInstance(proc_items, torch.Tensor)
        self.assertEqual(len1, len2)

    def test_transform_user(self):
        proc_user = self.preprocessor.preprocess_users(self.users[0])
        self.assertIsInstance(proc_user, torch.Tensor)
        self.assertEqual(proc_user.ndim, 0)

    def test_transform_item(self):
        proc_item = self.preprocessor.preprocess_items(self.items[0])
        self.assertIsInstance(proc_item, torch.Tensor)
        self.assertEqual(proc_item.ndim, 0)

    def test_transform_x(self):
        proc_users, proc_items = self.preprocessor.preprocess_x(self.features)
        self.assertIsInstance(proc_users, torch.Tensor)
        self.assertIsInstance(proc_items, torch.Tensor)
