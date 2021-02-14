import unittest
import os

from model_level import data_processing
from model_level.assistance import ModelAssistant
from saving.savers import StandardSaver
from saving.storages import LocalModelStorage
from ..helpers import std_objects, tests_config
config = tests_config.TestsConfig()


class TestStandardSaver(unittest.TestCase):
    def setUp(self):
        self.std_save_dir = config.save_dir

    def test_simple_model_save_load(self):
        assistant = std_objects.get_assistant()
        some_interacts = std_objects.get_interacts(30)
        assistant.update_with_interacts(some_interacts)

        storage = LocalModelStorage(self.std_save_dir)
        saver = StandardSaver(storage)
        model_name = saver.save(assistant)
        loaded_assist = saver.load(model_name)
        self.assertIsInstance(loaded_assist, ModelAssistant)
        self._assert_parameters_equal(assistant.get_model(), loaded_assist.get_model())

        old_convs = assistant.get_convs()
        loaded_convs = loaded_assist.get_convs()
        for old_converter, new_converter in zip(old_convs, loaded_convs):
            self._assert_equal_conv_ids(old_converter, new_converter)

    def test_multiple_models_loading(self):
        storage = LocalModelStorage(self.std_save_dir)
        saver = StandardSaver(storage)
        processor1 = data_processing.get_standard_processor()
        model1 = std_objects.get_model(2, 2, 2)
        assistant1 = ModelAssistant(model1, processor1)

        model2 = std_objects.get_model(5, 5, 5)
        processor2 = data_processing.get_standard_processor()
        assistant2 = ModelAssistant(model2, processor2)

        model1_name = saver.save(assistant1)
        model2_name = saver.save(assistant2)
        self.assertNotEqual(model1_name, model2_name, "different models have the same name")

        assistant1_loaded = saver.load(model1_name)
        assistant2_loaded = saver.load(model2_name)

        self._assert_parameters_equal(assistant1.get_model(), assistant1_loaded.get_model())
        self._assert_parameters_equal(assistant2.get_model(), assistant2_loaded.get_model())

        with self.assertRaises(ValueError):
            self._assert_parameters_equal(assistant1_loaded.get_model(), assistant2_loaded.get_model())

    def _assert_parameters_equal(self, model1, model2):
        for (old_param, new_param) in zip(model1.parameters(), model2.parameters()):
            old_param = old_param.detach().numpy()
            new_param = new_param.detach().numpy()
            if old_param.shape == new_param.shape:
                params_equal = (old_param == new_param).all()
                self.assertTrue(params_equal)
            else:
                raise ValueError("models parameters have different shape")

    def _assert_equal_conv_ids(self, conv1, conv2):
        ids1 = conv1.get_all_ids()
        ids2 = conv2.get_all_ids()
        self.assertEqual(list(ids1), list(ids2))

    def _delete_params_file(self, saver):
        params_path = saver.params_path
        os.remove(params_path)
