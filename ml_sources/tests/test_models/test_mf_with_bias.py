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
        
        self.preprocessor = preprocessing.TensorCreator(device)
        user_and_item = (343, 123)
        self.standard_single_inp = self.preprocessor.get_features_tensor(user_and_item)

        self.batchsize = 8
        users_items = [user_and_item for _ in range(self.batchsize)]
        self.standard_batch = self.preprocessor.get_features_tensor(users_items)

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
        input_with_new_user = self.preprocessor.get_features_tensor([new_user_id, some_item])

        model.add_users(1)
        pred = model(*input_with_new_user)
        self.assertEqual(pred.shape, torch.Size([1, 1]))

    def test_add_item(self):
        model = self._create_standard_model()
        new_item_id = self.standard_nitems # idxes start with 0
        some_user = 333
        input_with_new_item = self.preprocessor.get_features_tensor([some_user, new_item_id])

        model.add_items(1)
        pred = model(*input_with_new_item)
        self.assertEqual(pred.shape, torch.Size([1, 1]))

    def test_add_multiple_users(self):
        model = self._create_standard_model()
        new_ids = [self.standard_nusers + idx for idx in range(5)] # 5 users
        some_item = 333
        x = [[user_id, some_item] for user_id in new_ids]
        input_with_new_users = self.preprocessor.get_features_tensor(x)

        model.add_users(len(new_ids))
        pred = model(*input_with_new_users)

    def test_add_multiple_items(self):
        model = self._create_standard_model()
        new_item_ids = [self.standard_nitems + idx for idx in range(5)] # 5 users
        some_user = 333
        x = [[some_user, item_id] for item_id in new_item_ids]
        input_with_new_items = self.preprocessor.get_features_tensor(x)

        model.add_items(len(new_item_ids))
        pred = model(*input_with_new_items)

    def test_get_init_kwargs(self):
        model = self._create_standard_model()
        init_kwargs = model.get_init_kwargs()
        init_kwargs_structure_is_ok = "nusers" in init_kwargs and "nitems" in init_kwargs and "hidden_size" in init_kwargs
        self.assertTrue(init_kwargs_structure_is_ok)

    def test_adding_users_updated_init_kwargs(self):
        model = self._create_standard_model()
        old_nusers = model.get_init_kwargs()["nusers"]
        model.add_users(nusers=2)
        new_nusers = model.get_init_kwargs()["nusers"]
        self.assertEqual(old_nusers + 2, new_nusers)

    def test_saved_model_with_added_users_loads_with_correct_nusers(self):
        nb_added_users = 2
        model = self._create_standard_model()
        old_nusers = model.get_init_kwargs()["nusers"]
        model.add_users(nusers=nb_added_users)
        model_init_kwargs_before_saving = model.get_init_kwargs()
        new_nusers_before_saving = model_init_kwargs_before_saving["nusers"]
        self.assertEqual(old_nusers + nb_added_users, new_nusers_before_saving)

        saver = objects_creation.get_simple_saver(save_dir=config.save_dir)
        saver.save(model)

        expected_number_of_users = self.standard_nusers + nb_added_users
        new_model = saver.load(model)

        some_item = 0
        last_user = expected_number_of_users - 1
        non_existent_user = expected_number_of_users

        existent_user_inp = self.preprocessor.get_features_tensor([last_user, some_item])
        pred_for_exist_user = new_model(*existent_user_inp)

        non_existent_user_inp = self.preprocessor.get_features_tensor([non_existent_user, some_item])
        with self.assertRaises(IndexError):
            pred_non_exist = new_model(*non_existent_user_inp)

    def _create_standard_model(self):
        model = objects_creation.get_mf_model(self.standard_nusers,
                                              self.standard_nitems,
                                              hidden_size=self.standard_hidden_size)
        return model
