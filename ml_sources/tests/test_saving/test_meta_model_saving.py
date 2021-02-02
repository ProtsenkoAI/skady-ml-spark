# TODO
import torch
# import shutil
import os

from recsys_pipeline.data_transform.id_idx_conv import IdIdxConverter
from recsys_pipeline.models import mf_with_bias
from recsys_pipeline.saving import model_with_meta_and_ids_saving
from ..helpers import tests_config, objects_creation
from . import base_class
config = tests_config.TestsConfig()


class TestMetaModelSaving(base_class.TestSaving):
    def setUp(self):
        self.save_dir = config.save_dir
        self.standard_users = [123, 234]
        self.standard_items = [11, 22]
        self.item_conv = objects_creation.get_id_converter(*self.standard_items)
        self.user_conv = objects_creation.get_id_converter(*self.standard_users)

    def test_simple_model_saving(self):
        saver = model_with_meta_and_ids_saving.ModelAndIdsSaver(model_name="simple_model_104_104_20",
                                                                save_dir=self.save_dir)
        model, loaded_model, (saved_user_conv, saved_item_conv) = self._save_load_standard_model(saver, 10**4, 10**4, 20)
        self._assert_parameters_equal(model, loaded_model)
        self.assertIsInstance(saved_user_conv, IdIdxConverter)
        self.assertIsInstance(saved_item_conv, IdIdxConverter)

    def test_model_saving_with_custom_path_and_prefix(self):
        saver = model_with_meta_and_ids_saving.ModelAndIdsSaver(model_name="model", save_dir=self.save_dir,
                                                                params_file_name="params_aaa.json",
                                                                model_file_postfix="_omg_weights")
        model, loaded_model, (saved_user_conv, saved_item_conv) = self._save_load_standard_model(saver, 10, 10, 10)

        self._assert_parameters_equal(model, loaded_model)
        self.assertIsInstance(saved_user_conv, IdIdxConverter)
        self.assertIsInstance(saved_item_conv, IdIdxConverter)
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir, "params_aaa.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir, "model_omg_weights.pt")))

    def test_saving_loading_multiple_models(self):
        saver1 = model_with_meta_and_ids_saving.ModelAndIdsSaver(model_name="model1", save_dir=self.save_dir)
        saver2 = model_with_meta_and_ids_saving.ModelAndIdsSaver(model_name="model2", save_dir=self.save_dir)
        model1_kwargs = {"nusers": 1, "nitems": 1, "hidden_size": 1}
        model1 = mf_with_bias.MFWithBiasModel(**model1_kwargs)

        model2_kwargs = {"nusers": 2, "nitems": 2, "hidden_size": 2}
        model2 = mf_with_bias.MFWithBiasModel(**model2_kwargs)

        saver1.save(model1, self.user_conv, self.item_conv)
        saver2.save(model2, self.user_conv, self.item_conv)

        loaded_model1, (saved_user_conv, saved_item_conv) = saver1.load()
        self._assert_parameters_equal(model1, loaded_model1)
        for conv in [saved_user_conv, saved_item_conv]:
            self.assertIsInstance(conv, IdIdxConverter)

    def test_check_model_exists(self):
        saver = model_with_meta_and_ids_saving.ModelAndIdsSaver(model_name="model_aabbcc",
                                                                save_dir=self.save_dir,
                                                                params_file_name="params_for_exists_test.json")
        non_existing_model_result = saver.check_model_exists()

        model_kwargs = {"nusers": 2, "nitems": 2, "hidden_size": 2}
        model = mf_with_bias.MFWithBiasModel(**model_kwargs)
        saver.save(model, self.user_conv, self.item_conv)

        existing_model_result = saver.check_model_exists()

        self.assertEqual(non_existing_model_result, False)
        self.assertEqual(existing_model_result, True)

        self._delete_params_file(saver)

    def test_that_loaded_ids_are_in_same_order_as_saved(self):
        src_user_ids = self.standard_users
        src_item_ids = self.standard_items
        src_user_indexes = self.user_conv.get_idxs(*src_user_ids)
        src_item_indexes = self.item_conv.get_idxs(*src_item_ids)

        saver = model_with_meta_and_ids_saving.ModelAndIdsSaver(model_name="idk_some_model",
                                                                save_dir=self.save_dir,
                                                                params_file_name="params_for_ids_saving_check.json")
        _, _, (loaded_user_conv, loaded_item_conv) = self._save_load_standard_model(saver)

        loaded_user_indexes = loaded_user_conv.get_idxs(*src_user_ids)
        loaded_item_indexes = loaded_item_conv.get_idxs(*src_item_ids)

        pairs_must_be_equal = [(src_user_indexes, loaded_user_indexes),
                               (src_item_indexes, loaded_item_indexes)]

        for src, loaded in pairs_must_be_equal:
            self.assertEqual(src, loaded, "Some ids/idxs were changed while saving/loading.")

    def _save_load_standard_model(self, saver, nusers=50, nitems=50, hidden_size=20):
        model_kwargs = {"nusers": nusers, "nitems": nitems, "hidden_size": hidden_size}
        model = mf_with_bias.MFWithBiasModel(**model_kwargs)
        saver.save(model, self.user_conv, self.item_conv)

        loaded_model, (saved_user_conv, saved_item_conv) = saver.load()
        return model, loaded_model, (saved_user_conv, saved_item_conv)

    def _delete_params_file(self, saver):
        params_path = saver.params_path
        os.remove(params_path)
