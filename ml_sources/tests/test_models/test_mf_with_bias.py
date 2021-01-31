import unittest
import pandas as pd
import torch

from recsys_pipeline.models import mf_with_bias
from recsys_pipeline.data_transform import preprocessing
from ..helpers import tests_config, objects_creation
config = tests_config.TestsConfig()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestMFWithBias(unittest.TestCase):
    def setUp(self):
        self.standard_nusers = 5000
        self.standard_nitems = 5000
        self.standard_hidden_size = 50
        
        self.preprocessor = preprocessing.DataPreprocessor(device)
        user_and_item = (343, 123)
        self.standard_single_inp = self.preprocessor.preprocess_x(user_and_item)

        self.batchsize = 8
        users_items = [user_and_item for _ in range(self.batchsize)]
        self.standard_batch = self.preprocessor.preprocess_x(users_items)

    def test_creating_model(self):
        model = self._create_standard_model()

    def test_predict_batch(self):
        model = self._create_standard_model()
        pred = model.forward(*self.standard_batch)

        pred_batch_size = pred.shape[0]
        inp_batch_size = len(self.standard_batch[0])

        self.assertEqual(pred_batch_size, inp_batch_size)
        self.assertIsInstance(pred, torch.Tensor)

    def test_predict_single(self):
        model = self._create_standard_model()
        pred = model.forward(*self.standard_single_inp)
        self.assertEqual(pred.shape, torch.Size([1, 1]))

    def test_predict_one_user_many_items(self):
        model = self._create_standard_model()
        single_user = self.standard_single_inp[0]
        many_items = self.standard_batch[1]
        print("one to many: ", single_user, many_items)
        print(single_user.shape, many_items.shape)

        pred = model.forward(single_user, many_items)
        self.assertEqual(pred.shape, torch.Size([self.batchsize, 1]))

    def test_add_user(self):
        model = self._create_standard_model()
        new_user_id = self.standard_nusers # idxes start with 0
        some_item = 333
        input_with_new_user = self.preprocessor.preprocess_x([new_user_id, some_item])

        model.add_users(1)
        pred = model(*input_with_new_user)
        self.assertEqual(pred.shape, torch.Size([1, 1]))

    def test_add_item(self):
        model = self._create_standard_model()
        new_item_id = self.standard_nitems # idxes start with 0
        some_user = 333
        input_with_new_item = self.preprocessor.preprocess_x([some_user, new_item_id])

        model.add_items(1)
        pred = model(*input_with_new_item)
        self.assertEqual(pred.shape, torch.Size([1, 1]))

    def test_add_multiple_users(self):
        model = self._create_standard_model()
        new_ids = [self.standard_nusers + idx for idx in range(5)] # 5 users
        some_item = 333
        x = [[user_id, some_item] for user_id in new_ids]
        input_with_new_users = self.preprocessor.preprocess_x(x)

        model.add_users(len(new_ids))
        pred = model(*input_with_new_users)

    def test_add_multiple_items(self):
        model = self._create_standard_model()
        new_item_ids = [self.standard_nitems + idx for idx in range(5)] # 5 users
        some_user = 333
        x = [[some_user, item_id] for item_id in new_item_ids]
        input_with_new_items = self.preprocessor.preprocess_x(x)

        model.add_items(len(new_item_ids))
        pred = model(*input_with_new_items)

    def _create_standard_model(self):
        model = objects_creation.get_mf_model(self.standard_nusers,
                                              self.standard_nitems,
                                              hidden_size=self.standard_hidden_size)
        return model
