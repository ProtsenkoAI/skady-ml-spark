import unittest
import math

from high_level_managing.train_pipeline import TrainPipelineManager
from high_level_managing.train_pipeline_scheduler import TrainPipelineScheduler
from model_level.assistance import ModelAssistant

from ..helpers import std_objects


class TestTrainPipelineManager(unittest.TestCase):
    def test_model_saving_loading(self):
        manager = std_objects.get_train_pipeline_manager()
        eval_interacts = std_objects.get_interacts()
        eval_res = manager.run(std_objects.get_interacts(), eval_interacts)
        loaded_assistant = manager.load_best()
        self.assertIsInstance(loaded_assistant, ModelAssistant)

    def test_eval_every_epoch_break_because_of_steps(self):
        max_steps = 10
        trainer = std_objects.get_trainer()
        steps_in_epoch = len( std_objects.get_dataloader())
        scheduler = TrainPipelineScheduler(steps_in_epoch, max_steps=max_steps, max_epochs=5, eval_strategy="epochs")
        manager = TrainPipelineManager(std_objects.get_assistant(), trainer,
                                        std_objects.get_validator(),  std_objects.get_standard_saver(),
                                       scheduler)

        eval_interacts = std_objects.get_interacts()
        eval_res = manager.run(std_objects.get_interacts(), eval_interacts)
        epochs_in_max_steps = math.ceil(max_steps / steps_in_epoch)
        self.assertEqual(len(eval_res), epochs_in_max_steps)

    def test_eval_every_nsteps_break_because_of_steps(self):
        max_steps = 10
        trainer =  std_objects.get_trainer()
        steps_in_epoch = len( std_objects.get_dataloader())
        scheduler = TrainPipelineScheduler(steps_in_epoch, max_steps=max_steps, max_epochs=5,
                                           eval_strategy="steps", nsteps=8)
        manager = TrainPipelineManager( std_objects.get_assistant(), trainer,
                                        std_objects.get_validator(),  std_objects.get_standard_saver(), scheduler)

        eval_interacts = std_objects.get_interacts()
        eval_res = manager.run(std_objects.get_interacts(), eval_interacts)
        self.assertEqual(len(eval_res), 2)

    def test_eval_every_epoch_break_because_of_epochs(self):
        trainer =  std_objects.get_trainer()
        steps_in_epoch = len( std_objects.get_dataloader())
        scheduler = TrainPipelineScheduler(steps_in_epoch, max_steps=9999, max_epochs=5, eval_strategy="epochs")
        manager = TrainPipelineManager( std_objects.get_assistant(), trainer,
                                        std_objects.get_validator(),  std_objects.get_standard_saver(),
                                       scheduler)

        eval_interacts = std_objects.get_interacts()
        eval_res = manager.run(std_objects.get_interacts(), eval_interacts)
        self.assertEqual(len(eval_res), 5)

    def test_eval_every_nsteps_break_because_of_epochs(self):
        max_epochs = 5
        trainer =  std_objects.get_trainer()
        steps_in_epoch = len( std_objects.get_dataloader())
        scheduler = TrainPipelineScheduler(steps_in_epoch, max_steps=9999, max_epochs=max_epochs,
                                           eval_strategy="steps", nsteps=5)
        manager = TrainPipelineManager(std_objects.get_assistant(), trainer,
                                        std_objects.get_validator(),  std_objects.get_standard_saver(), scheduler)

        eval_interacts = std_objects.get_interacts()
        eval_res = manager.run(std_objects.get_interacts(), eval_interacts)
        self.assertTrue(len(eval_res), max_epochs)

    def test_exceed_stopping_patience(self):
        """We can't manually enforce eval value to decrease (to induce stopping because of the stopping patience
        exceeding. So we'll just set stopping patience, as well as max_steps and'll check that the manager
        will not break at functions devoted for stopping_patience checking
        """
        trainer =  std_objects.get_trainer()
        steps_in_epoch = len( std_objects.get_dataloader())
        scheduler = TrainPipelineScheduler(steps_in_epoch, eval_strategy="steps", nsteps=5, stop_patience=2)
        manager = TrainPipelineManager( std_objects.get_assistant(), trainer,
                                        std_objects.get_validator(), std_objects.get_standard_saver(), scheduler)

        eval_interacts = std_objects.get_interacts()
        eval_res = manager.run(std_objects.get_interacts(), eval_interacts)
        pass

    def test_raises_error_if_eval_strategy_steps_but_no_nsteps_provided(self):
        trainer = std_objects.get_trainer()
        train_dataset = std_objects.get_dataloader()
        steps_in_epoch = len(train_dataset)
        scheduler = TrainPipelineScheduler(steps_in_epoch, eval_strategy="steps")
        manager = TrainPipelineManager( std_objects.get_assistant(), trainer,
                                        std_objects.get_validator(), std_objects.get_standard_saver(), scheduler)
        eval_interacts = std_objects.get_interacts()
        with self.assertRaises(ValueError):
            eval_res = manager.run(std_objects.get_interacts(), eval_interacts)
