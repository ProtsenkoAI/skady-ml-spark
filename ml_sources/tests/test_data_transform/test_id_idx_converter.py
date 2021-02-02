import unittest
import pandas as pd

from recsys_pipeline.data_transform import id_idx_conv
from ..helpers import tests_config
config = tests_config.TestsConfig()

class TestIdIdxConverter(unittest.TestCase):
    def test_initializing_with_ids(self):
        init_ids = [111, 222, 333]
        converter = id_idx_conv.IdIdxConverter(*init_ids)

    def test_convert_id_to_idx(self):
        single_id = 123
        converter = id_idx_conv.IdIdxConverter()
        idx = converter.add_ids_get_idxs(single_id)

    def test_convert_ids_to_idxs(self):
        multiple_ids = [123, 234, 345]
        converter = id_idx_conv.IdIdxConverter()
        indexes = converter.add_ids_get_idxs(*multiple_ids)

    def test_convert_idx_to_id(self):
        src_elem = 10**5
        converter = id_idx_conv.IdIdxConverter(src_elem)
        elem_from_converter = converter.get_ids(0)
        self.assertEqual(src_elem, elem_from_converter)

    def test_idx_values_after_initing(self):
        init_ids = [111, 222, 333]
        converter = id_idx_conv.IdIdxConverter(*init_ids)
        init_idxs = converter.get_idxs(*init_ids)
        self.assertEqual(min(init_idxs), 0, "Idxs don't stort with 0")
        self.assertEqual(max(init_idxs), 2, "Idxs aren't consecutive")

    def test_idx_values_after_addiing(self):
        init_ids = [111, 222, 333]
        converter = id_idx_conv.IdIdxConverter()
        idxs = converter.add_ids_get_idxs(*init_ids)
        new_elems = [444, 555]
        new_idxs = converter.add_ids_get_idxs(*new_elems)
        self.assertEqual(max(idxs), min(new_idxs) - 1, "Idxs aren't consecutive")

    def test_id_inverse_transform(self):
        elem_id = 456
        converter = id_idx_conv.IdIdxConverter()
        idx = converter.add_ids_get_idxs(elem_id)
        converted_id = converter.get_ids(idx)
        self.assertEqual(elem_id, converted_id, "Inverse transform error")

    def test_count_unknown(self):
        elems = [1, 2, 3]
        converter = id_idx_conv.IdIdxConverter(*elems)
        new_elems = [2, 3, 4, 5, 6]
        num_unknown = converter.count_unknown(*new_elems)
        self.assertEqual(num_unknown, 3)

    def test_get_all_ids(self):
        elems = [1, 2, 3]
        converter = id_idx_conv.IdIdxConverter(*elems)
        new_elems = [4, 5, 6]
        converter.add_ids_get_idxs(*new_elems)

        all_saved_ids = converter.get_all_ids()
        self.assertEqual(list(all_saved_ids), [1, 2, 3, 4, 5, 6])

    def test_idx_inverse_transform(self):
        elem_id = 456
        src_idx = 0
        converter = id_idx_conv.IdIdxConverter(elem_id)
        converted_id = converter.get_ids(src_idx)
        converted_idx_from_id = converter.get_idxs(converted_id)

        self.assertEqual(src_idx, converted_idx_from_id, "Inverse transform error")
