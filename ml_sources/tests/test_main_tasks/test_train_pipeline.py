# NOTE: at the moment tests are based at hardcoded trainer.get_dataset_len() -> 10, so when 'll write normal code
# it can break here.

import unittest
import math

from recsys_pipeline.main_tasks.train_pipeline import TrainPipelineManager
from recsys_pipeline.main_tasks.train_pipeline_scheduler import TrainPipelineScheduler
from recsys_pipeline.model_level.assistance import ModelAssistant
from ..helpers.objs_pool import ObjsPool
objs_pool = ObjsPool()


class TestTrainPipelineManager(unittest.TestCase):
    def test_model_saving_loading(self):
        manager = objs_pool.train_pipeline_manager
        dataloader = objs_pool.train_dataloader
        eval_res = manager.run(dataloader)
        loaded_assistant = manager.load_best()
        self.assertIsInstance(loaded_assistant, ModelAssistant)

    def test_eval_every_epoch_break_because_of_steps(self):
        max_steps = 10
        trainer = objs_pool.trainer
        steps_in_epoch = len(objs_pool.train_dataloader)
        scheduler = TrainPipelineScheduler(steps_in_epoch, max_steps=max_steps, max_epochs=5, eval_strategy="epochs")
        manager = TrainPipelineManager(objs_pool.assistant, trainer,
                                       objs_pool.validator, objs_pool.standard_saver,
                                       scheduler)

        eval_res = manager.run(objs_pool.train_dataloader)
        epochs_in_max_steps = math.ceil(max_steps / steps_in_epoch)
        self.assertEqual(len(eval_res), epochs_in_max_steps)

    def test_eval_every_nsteps_break_because_of_steps(self):
        max_steps = 10
        trainer = objs_pool.trainer
        steps_in_epoch = len(objs_pool.train_dataloader)
        scheduler = TrainPipelineScheduler(steps_in_epoch, max_steps=max_steps, max_epochs=5,
                                           eval_strategy="steps", nsteps=8)
        manager = TrainPipelineManager(objs_pool.assistant, trainer,
                                       objs_pool.validator, objs_pool.standard_saver, scheduler)

        eval_res = manager.run(objs_pool.train_dataloader)
        self.assertEqual(len(eval_res), 2)

    def test_eval_every_epoch_break_because_of_epochs(self):
        trainer = objs_pool.trainer
        steps_in_epoch = len(objs_pool.train_dataloader)
        scheduler = TrainPipelineScheduler(steps_in_epoch, max_steps=9999, max_epochs=5, eval_strategy="epochs")
        manager = TrainPipelineManager(objs_pool.assistant, trainer,
                                       objs_pool.validator, objs_pool.standard_saver,
                                       scheduler)

        eval_res = manager.run(objs_pool.train_dataloader)
        self.assertEqual(len(eval_res), 5)

    def test_eval_every_nsteps_break_because_of_epochs(self):
        max_epochs = 5
        trainer = objs_pool.trainer
        steps_in_epoch = len(objs_pool.train_dataloader)
        scheduler = TrainPipelineScheduler(steps_in_epoch, max_steps=9999, max_epochs=max_epochs,
                                           eval_strategy="steps", nsteps=5)
        manager = TrainPipelineManager(objs_pool.assistant, trainer,
                                       objs_pool.validator, objs_pool.standard_saver, scheduler)

        eval_res = manager.run(objs_pool.train_dataloader)
        self.assertTrue(len(eval_res), max_epochs)

    def test_exceed_stopping_patience(self):
        """We can't manually enforce eval value to decrease (to induce stopping because of the stopping patience
        exceeding. So we'll just set stopping patience, as well as max_steps and'll check that the manager
        will not break at functions devoted for stopping_patience checking
        """
        trainer = objs_pool.trainer
        steps_in_epoch = len(objs_pool.train_dataloader)
        scheduler = TrainPipelineScheduler(steps_in_epoch, eval_strategy="steps", nsteps=5, stop_patience=2)
        manager = TrainPipelineManager(objs_pool.assistant, trainer,
                                       objs_pool.validator, objs_pool.standard_saver, scheduler)

        eval_res = manager.run(objs_pool.train_dataloader)
        pass

    def test_raises_error_if_eval_strategy_steps_but_no_nsteps_provided(self):
        trainer = objs_pool.trainer
        train_dataset = objs_pool.train_dataloader
        steps_in_epoch = len(train_dataset)
        scheduler = TrainPipelineScheduler(steps_in_epoch, eval_strategy="steps")
        manager = TrainPipelineManager(objs_pool.assistant, trainer,
                                       objs_pool.validator, objs_pool.standard_saver, scheduler)
        with self.assertRaises(ValueError):
            manager.run(train_dataset)