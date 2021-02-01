# TODO
import copy
import os
# from recsys_pipeline.models import mf_with_bias
from recsys_pipeline.saving import model_state_dict_saving
from ..helpers import tests_config, objects_creation
from . import base_class
config = tests_config.TestsConfig()


class TestModelStateDictSaver(base_class.TestSaving):
    def setUp(self):
        self.save_dir = config.save_dir
        self.nusers = 20
        self.nitems = 20
        self.hidden_size = 5
        self.standard_model = objects_creation.get_mf_model(self.nusers, self.nitems, hidden_size=self.hidden_size)

    def test_save_load(self):
        saver = self._create_saver()
        self._save_load_assert_parameters_equal(saver, self.standard_model)

    def test_save_load_custom_path(self):
        custom_save_dir = os.path.join(config.save_dir, "lul_folded_dir_for_custom_saving_path/")
        saver = self._create_saver(save_dir=custom_save_dir)
        self._save_load_assert_parameters_equal(saver, self.standard_model)
        self.assertTrue(os.path.isdir(custom_save_dir))

    def test_save_load_custom_model_file_name(self):
        custom_model_name = "custom_weights_name"
        saver = self._create_saver(model_file_name=custom_model_name)
        self._save_load_assert_parameters_equal(saver, self.standard_model)

    def test_save_load_multiple_models(self):
        model1_name = "model1_weights"
        saver1 = self._create_saver(model_file_name=model1_name)
        self._save_load_assert_parameters_equal(saver1, self.standard_model)

        model2 = objects_creation.get_mf_model(self.nusers + 5, self.nitems + 5, hidden_size=self.hidden_size + 5)
        model2_name = "model2_weights)"
        saver2 = self._create_saver(model_file_name=model2_name)
        self._save_load_assert_parameters_equal(saver2, model2)

    def _save_load_assert_parameters_equal(self, saver, model):
        old_model = copy.deepcopy(self.standard_model)
        state_dict = self.standard_model.state_dict()
        saver.save(state_dict)
        loaded_state_dict = saver.load()
        self.standard_model.load_state_dict(loaded_state_dict)
        new_model = self.standard_model

        self._assert_parameters_equal(old_model, new_model)

    def _create_saver(self, save_dir=config.save_dir, model_file_name="model_weights"):
        return model_state_dict_saving.ModelStateDictSaver(save_dir, model_file_name)