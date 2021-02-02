# TODO
import unittest
from torch.utils import data as torch_data

from recsys_pipeline.data import loader_factories, datasets
from ..helpers import tests_config, objects_creation
config = tests_config.TestsConfig()


class TestStandardLoaderBuilder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 12
        self.nrows = 100

    def test_init_and_call(self):
        loader = self._build_standard_loader()
        is_dataloader = isinstance(loader, torch_data.DataLoader)
        self.assertTrue(is_dataloader, "StandardLoaderBuilder doesn't produce DataLoader")

    def test_batchsize_is_correct(self):
        loader = self._build_standard_loader()
        batch = next(iter(loader))
        *features, labels = batch
        batch_size_of_labels = labels.shape[0]
        self.assertEqual(batch_size_of_labels, self.batch_size, "batch size is incorrect")

    def _build_standard_loader(self):
        interacts = objects_creation.get_interacts(nrows=self.nrows)
        builder = loader_factories.StandardLoaderBuilder(batch_size=self.batch_size)
        loader = builder(interacts)
        return loader
