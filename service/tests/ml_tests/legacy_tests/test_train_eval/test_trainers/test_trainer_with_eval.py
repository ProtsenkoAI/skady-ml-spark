import unittest

from train import EvalTrainer
from model import LocalModelManager

from helpers import std_objects


class TestTrainPipelineManager(unittest.TestCase):
    def test_model_saving_loading(self):
        manager = std_objects.get_eval_trainer(steps_between_evals=1)
        eval_res = manager.fit(std_objects.get_assistant(), std_objects.get_interacts(), max_step=1)
        loaded_assistant = manager.load_best()
        self.assertIsInstance(loaded_assistant, LocalModelManager)

    def test_break_because_of_steps(self):
        manager = EvalTrainer(std_objects.get_validator(), std_objects.get_dataloader_builder(),
                              std_objects.get_standard_saver(), steps_between_evals=5)

        manager.fit(std_objects.get_assistant(), std_objects.get_interacts(), max_step=10)
        eval_res = manager.get_val_results()
        self.assertEqual(len(eval_res), 2)

    def test_break_because_of_epochs(self):
        manager = EvalTrainer(std_objects.get_validator(), std_objects.get_dataloader_builder(),
                                        std_objects.get_standard_saver())
        manager.fit(std_objects.get_assistant(), std_objects.get_interacts(), max_epoch=2)

    def test_exceed_stopping_patience(self):
        """We can't manually enforce eval value to decrease (to induce stopping because of the stopping patience
        exceeding. So we'll just set stopping patience, as well as max_steps and'll check that the manager
        will not break at functions devoted for stopping_patience checking
        """
        manager = EvalTrainer(std_objects.get_validator(), std_objects.get_dataloader_builder(),
                              std_objects.get_standard_saver())

        manager.fit(std_objects.get_assistant(), std_objects.get_interacts(), stop_patience=2, max_step=3)
        pass
