import unittest

from .storages_testing_help import StorageTestingHelper
from storages.local_model_storage import LocalModelStorage
from helpers import tests_config
config = tests_config.TestsConfig()


class TestLocalModelStorage(unittest.TestCase):
    def setUp(self):
        self.storage = LocalModelStorage(save_dir=config.save_dir, weights_postfix="_lul_weights_file",
                                 meta_file_name="info_about_saved_models")
        self.helper = StorageTestingHelper()

    def test_check_unsaved_model_does_not_exist_then_save_model_and_check_it_exists(self):
        self.helper.test_check_unsaved_model_does_not_exist_then_save_model_and_check_it_exists(self, self.storage)

    def test_save_check_exists_then_delete_check_does_not_exists(self):
        self.helper.test_save_check_exists_then_delete_check_does_not_exists(self, self.storage)
