import unittest
import numpy as np

from model.manage.data_processing import IdIdxConv
from helpers import tests_config

config = tests_config.TestsConfig()


class TestIdIdxConv(unittest.TestCase):
    def test_add_ids_get_same_ids(self):
        src_ids = [1, 2, 3]
        conv = IdIdxConv(src_ids)
        loaded_ids = conv.get_all_ids()
        self.assertEqual(src_ids, loaded_ids)

    def test_add_ids_multiple_times(self):
        conv = IdIdxConv()
        added_ids_lists = np.arange(9).reshape((3, 3))
        for ids_lst in added_ids_lists:
            conv.add_ids(*ids_lst)
        all_ids = conv.get_all_ids()
        self.assertEqual(len(all_ids), 3 * 3)

    def test_add_duplicate_ids_doesnt_add_elements(self):
        conv = IdIdxConv()
        for _ in range(5):
            conv.add_ids(1, 2)
        all_idxs = conv.get_all_idxs()
        self.assertEqual(len(all_idxs), 2)

    def reverse_ids_to_idxs_to_ids(self):
        conv = IdIdxConv()
        ids = [1, 2, 3]
        conv.add_ids(*ids)
        idxs = conv.get_idxs(*ids)
        ids_from_idxs = conv.get_ids(*idxs)
        self.assertEqual(ids_from_idxs, ids)

    def test_all_ids_and_idxs(self):
        conv = IdIdxConv()
        ids = [i for i in range(5)]
        conv.add_ids(*ids)

        all_ids, all_idxs = conv.get_all_ids(), conv.get_all_idxs()
        self.assertEqual(len(all_ids), len(ids))
        self.assertEqual(len(all_idxs), len(ids))

    def test_dump(self):
        added_ids = [333, 22, 1]
        conv = IdIdxConv(added_ids)
        dumped_conv = conv.dump()
        self.assertEqual(dumped_conv, added_ids)

    def test_dump_then_load(self):
        added_ids = [333, 22, 1]
        conv = IdIdxConv(added_ids)
        old_ids = conv.get_all_ids()
        dumped_conv = conv.dump()
        loaded_conv = IdIdxConv.load(dumped_conv)
        loaded_ids = loaded_conv.get_all_ids()
        self.assertEqual(old_ids, loaded_ids)

    def test_single_id_idx_support(self):
        conv = IdIdxConv()
        conv.add_ids(3)
        first_elem_id = conv.get_ids(0)
        idx_of_first_elem = conv.get_idxs(3)
        unknown = conv.count_unknown(77)

        self.assertEqual(first_elem_id, [3])
        self.assertEqual(idx_of_first_elem, [0])
        self.assertEqual(unknown, 1)
