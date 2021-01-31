# TODO
import unittest
import torch
# import shutil
import os

from recsys_pipeline.models import mf_with_bias
from recsys_pipeline.saving import meta_model_saving
from ..helpers import tests_config, objects_creation
config = tests_config.TestsConfig()


class TestMetaModelSaver(unittest.TestCase):
    def setUp(self):
        self.save_dir = config.save_dir
        self.standard_users = [123, 234]
        self.standard_items = [11, 22]
        self.item_conv = objects_creation.get_id_converter(*self.standard_items)
        self.user_conv = objects_creation.get_id_converter(*self.standard_users)

    def test_simple_model_saving(self):
        saver = meta_model_saving.MetaModelSaver(save_dir=self.save_dir)
        model, loaded_model, (saved_user_ids, saved_item_ids) = self._save_load_standard_model(saver,
                                                             "simple_model_104_104_20", 10**4, 10**4, 20)
        self._assert_parameters_equal(model, loaded_model)
        self.assertIsInstance(saved_user_ids, list)
        self.assertIsInstance(saved_item_ids, list)

    def test_model_saving_with_custom_path_and_prefix(self):
        saver = meta_model_saving.MetaModelSaver(save_dir=self.save_dir, params_file_name="params_aaa.json",
                                            model_file_postfix="_omg_weights")
        model, loaded_model, (saved_user_ids, saved_item_ids) = self._save_load_standard_model(saver, "model", 10, 10, 10)

        self._assert_parameters_equal(model, loaded_model)
        self.assertIsInstance(saved_user_ids, list)
        self.assertIsInstance(saved_item_ids, list)
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir, "params_aaa.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir, "model_omg_weights.pt")))

    def test_saving_loading_multiple_models(self):
        saver = meta_model_saving.MetaModelSaver(save_dir=self.save_dir)
        model1_kwargs = {"nusers": 1, "nitems": 1, "hidden_size": 1}
        model1 = mf_with_bias.MFWithBiasModel(**model1_kwargs)

        model2_kwargs = {"nusers": 2, "nitems": 2, "hidden_size": 2}
        model2 = mf_with_bias.MFWithBiasModel(**model2_kwargs)

        saver.save("model1", model1.state_dict(), self.user_conv.get_all_ids(), self.item_conv.get_all_ids(),
                   meta_info=model1_kwargs)
        saver.save("model2", model2.state_dict(), self.user_conv.get_all_ids(), self.item_conv.get_all_ids(),
                   meta_info=model2_kwargs)

        loaded_model1, (saved_user_ids, saved_item_ids) = saver.load("model1")
        self._assert_parameters_equal(model1, loaded_model1)

    def test_check_model_exists(self):
        model_name = "model_aabbcc"
        saver = meta_model_saving.MetaModelSaver(save_dir=self.save_dir, params_file_name="params_for_exists_test.json")
        non_existing_model_result = saver.check_model_exists(model_name)

        model_kwargs = {"nusers": 2, "nitems": 2, "hidden_size": 2}
        model = mf_with_bias.MFWithBiasModel(**model_kwargs)
        saver.save(model_name, model.state_dict(), self.user_conv.get_all_ids(), self.item_conv.get_all_ids(),
                   model_kwargs)

        existing_model_result = saver.check_model_exists(model_name)

        self.assertEqual(non_existing_model_result, False)
        self.assertEqual(existing_model_result, True)

        self._delete_params_file(saver)

    def _assert_parameters_equal(self, model1, model2):
        for (old_param, new_param) in zip(model1.parameters(), model2.parameters()):
            old_param = old_param.detach().numpy()
            new_param = new_param.detach().numpy()
            if old_param.shape == new_param.shape:
                params_equal = (old_param == new_param).all()
                self.assertTrue(params_equal)

    def _save_load_standard_model(self, saver, model_name, nusers=50, nitems=50, hidden_size=20):
        model_kwargs = {"nusers": nusers, "nitems": nitems, "hidden_size": hidden_size}
        model = mf_with_bias.MFWithBiasModel(**model_kwargs)
        saver.save(model_name, model.state_dict(), self.user_conv.get_all_ids(), self.item_conv.get_all_ids(), meta_info=model_kwargs)

        loaded_model, (saved_user_ids, saved_item_ids) = saver.load(model_name)
        return model, loaded_model, (saved_user_ids, saved_item_ids)

    def _delete_params_file(self, saver):
        params_path = saver.params_path
        os.remove(params_path)


    # def tearDown(self, dir_path):
    #     # clean files
    #     shutil.rmtree(self.save_dir)
