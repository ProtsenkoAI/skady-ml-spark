import unittest
import os

from model_level import data_processing
from saving.savers import StandardSaver
from storages import LocalModelStorage
from helpers import tests_config, std_objects

config = tests_config.TestsConfig()


class TestStandardSaver(unittest.TestCase):
    def setUp(self):
        self.std_save_dir = config.save_dir

    def test_simple_model_save_load(self):
        model = std_objects.get_model()
        processor = std_objects.get_standard_processor()

        storage = LocalModelStorage(self.std_save_dir)
        saver = StandardSaver(storage)
        model_name = saver.save(model, processor)
        source_model_class = type(model)
        source_proc_class = type(processor)
        del model, processor
        loaded_model, loaded_processor = saver.load(model_name)
        self.assertIsInstance(loaded_model, source_model_class)
        self.assertIsInstance(loaded_processor, source_proc_class)

    def test_multiple_models_loading(self):
        storage = LocalModelStorage(self.std_save_dir)
        saver = StandardSaver(storage)
        processor1 = data_processing.get_standard_processor()
        model1 = std_objects.get_model(2, 2, 2)

        model2 = std_objects.get_model(5, 5, 5)
        processor2 = data_processing.get_standard_processor()

        model1_name = saver.save(model1, processor1)
        model2_name = saver.save(model2, processor2)
        self.assertNotEqual(model1_name, model2_name, "different models have the same name")

        model1_loaded, proc1_loaded = saver.load(model1_name)
        model2_loaded, proc2_loaded = saver.load(model2_name)

        self._assert_parameters_equal(model1, model1_loaded)
        self._assert_parameters_equal(model2, model2_loaded)

        with self.assertRaises(ValueError):
            self._assert_parameters_equal(model1_loaded, model2_loaded)

    def _assert_parameters_equal(self, model1, model2):
        for (old_param, new_param) in zip(model1.parameters(), model2.parameters()):
            old_param = old_param.detach().numpy()
            new_param = new_param.detach().numpy()
            if old_param.shape == new_param.shape:
                params_equal = (old_param == new_param).all()
                self.assertTrue(params_equal)
            else:
                raise ValueError("models parameters have different shape")

    def _delete_params_file(self, saver):
        params_path = saver.params_path
        os.remove(params_path)
