# import unittest
# import os
# import copy
#
# from ..helpers import std_objects, tests_config
# config = tests_config.TestsConfig()
#
#
# class TestTrainer(unittest.TestCase):
#     def test_fit_steps(self):
#         trainer = std_objects.get_trainer()
#         assistant = std_objects.get_assistant()
#         interacts = std_objects.get_interacts()
#         assistant.update_with_interacts(interacts)
#         trainer.fit(assistant, interacts, nsteps=5)
#
#     def test_fit_epochs(self):
#         trainer = std_objects.get_trainer()
#         assistant = std_objects.get_assistant()
#         interacts = std_objects.get_interacts()
#         assistant.update_with_interacts(interacts)
#         trainer.fit(assistant, interacts, nepochs=2)
#
#     def test_weights_are_updated_during_fit(self):
#         trainer = std_objects.get_trainer()
#         assistant = std_objects.get_assistant()
#         interacts = std_objects.get_interacts()
#         assistant.update_with_interacts(interacts)
#
#         weights_before = self._get_model_weights(assistant.get_model())
#         trainer.fit(assistant, interacts, nepochs=2)
#         weights_after = self._get_model_weights(assistant.get_model())
#
#         for old, new in zip(weights_before, weights_after):
#             old_and_new_are_identical = (old == new).all()
#             self.assertFalse(old_and_new_are_identical)
#
#     def _get_model_weights(self, model):
#         weights = []
#         for param in model.parameters():
#             param_weights = param.detach().numpy().copy()
#             weights.append(param_weights)
#
#         return weights