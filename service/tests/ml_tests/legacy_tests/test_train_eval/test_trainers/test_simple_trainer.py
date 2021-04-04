import unittest
import os
import copy

from helpers import std_objects, tests_config
config = tests_config.TestsConfig()


class TestTrainer(unittest.TestCase):
    def test_fit_steps(self):
        trainer = std_objects.get_simple_trainer()
        assistant = std_objects.get_assistant()
        interacts = std_objects.get_interacts()
        trainer.fit(assistant, interacts, max_step=5)

    def test_fit_epochs(self):
        trainer = std_objects.get_simple_trainer()
        assistant = std_objects.get_assistant()
        interacts = std_objects.get_interacts()
        trainer.fit(assistant, interacts, max_epoch=2)

    def test_weights_are_updated_during_fit(self):
        trainer = std_objects.get_simple_trainer()
        assistant = std_objects.get_assistant()
        interacts = std_objects.get_interacts()
        weights_before = self._get_model_weights(assistant.get_model_manager())
        trainer.fit(assistant, interacts, max_epoch=2)
        weights_after = self._get_model_weights(assistant.get_model_manager())

        for old, new in zip(weights_before, weights_after):
            old_new_compare = old == new
            if isinstance(old_new_compare, bool):
                continue  # have different shapes
            old_and_new_are_identical = (old == new).all()
            self.assertFalse(old_and_new_are_identical)

    def _get_model_weights(self, model):
        weights = []
        for param in model.parameters():
            param_weights = param.detach().numpy().copy()
            weights.append(param_weights)

        return weights