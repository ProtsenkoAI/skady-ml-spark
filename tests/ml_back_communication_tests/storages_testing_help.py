import unittest

from helpers import std_objects


class StorageTestingHelper:
    def __init__(self):
        model = std_objects.get_model()
        self.weights = model.state_dict()
        user_ids = [1, 2, 3]
        item_ids = [2, 3, 4]
        self.meta = {"some_model_info": {"nusers": 5, "lul, i mean": "so what"},
                     "user_conv": user_ids, "item_conv": item_ids}

    def test_check_unsaved_model_does_not_exist_then_save_model_and_check_it_exists(self, test_case: unittest.TestCase,
                                                                                    storage):
        check_unexisted_res = storage.check_model_exists("some_unexisted_model_name")
        test_case.assertFalse(check_unexisted_res)

        name = storage.save_weights_and_meta(self.weights, self.meta)
        check_existed_res = storage.check_model_exists(name)
        test_case.assertTrue(check_existed_res)

    def test_save_check_exists_then_delete_check_does_not_exists(self, test_case, storage):
        name = storage.save_weights_and_meta(self.weights, self.meta)
        saved_model_exists = storage.check_model_exists(name)
        test_case.assertTrue(saved_model_exists)
        storage.delete_model(name)
        deleted_model_exists = storage.check_model_exists(name)
        test_case.assertFalse(deleted_model_exists)
