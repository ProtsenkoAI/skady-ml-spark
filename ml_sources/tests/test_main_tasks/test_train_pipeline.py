# NOTE: at the moment tests are based at hardcoded trainer.get_dataset_len() -> 10, so when 'll write normal code
# it can break here.

import unittest

from recsys_pipeline.main_tasks.train_pipeline import TrainPipelineManager
from recsys_pipeline.model_level.assistance import ModelAssistant
from ..helpers.objs_pool import ObjsPool
objs_pool = ObjsPool()


class TestTrainPipelineManager(unittest.TestCase):
    def test_model_saving_loading(self):
        manager = objs_pool.train_pipeline_manager
        eval_res = manager.run()
        loaded_assistant = manager.load_best()
        self.assertIsInstance(loaded_assistant, ModelAssistant)

    def test_eval_every_epoch_break_because_of_steps(self):
        max_steps = 10
        epochs_in_max_steps = 1
        manager = TrainPipelineManager(objs_pool.assistant, objs_pool.trainer,
                                       objs_pool.validator, objs_pool.weights_saver,
                                       max_steps=max_steps, max_epochs=5, eval_strategy="epochs")

        eval_res = manager.run()
        self.assertEqual(len(eval_res), epochs_in_max_steps)

    def test_eval_every_nsteps_break_because_of_steps(self):
        max_steps = 10
        manager = TrainPipelineManager(objs_pool.assistant, objs_pool.trainer,
                                       objs_pool.validator, objs_pool.weights_saver,
                                       max_steps=max_steps, max_epochs=5, eval_strategy="steps",
                                       nsteps=8)

        eval_res = manager.run()
        self.assertEqual(len(eval_res), 2)

    def test_eval_every_epoch_break_because_of_epochs(self):
        manager = TrainPipelineManager(objs_pool.assistant, objs_pool.trainer,
                                       objs_pool.validator, objs_pool.weights_saver,
                                       max_steps=9999, max_epochs=5, eval_strategy="epochs")

        eval_res = manager.run()
        self.assertEqual(len(eval_res), 5)

    def test_eval_every_nsteps_break_because_of_epochs(self):
        manager = TrainPipelineManager(objs_pool.assistant, objs_pool.trainer,
                                                 objs_pool.validator, objs_pool.weights_saver,
                                                 max_steps=9999, max_epochs=5, eval_strategy="steps", nsteps=5)

        eval_res = manager.run()
        self.assertEqual(len(eval_res), 10)

    def test_exceed_stopping_patience(self):
        """We can't manually enforce eval value to decrease (to induce stopping because of the stopping patience
        exceeding. So we'll just set stopping patience, as well as max_steps and'll check that the manager
        will not break at functions devoted for stopping_patience checking
        """
        manager = TrainPipelineManager(objs_pool.assistant, objs_pool.trainer,
                                       objs_pool.validator, objs_pool.weights_saver,
                                       eval_strategy="steps", nsteps=5,
                                       stop_patience=2)

        eval_res = manager.run()
        pass

    def test_raises_error_if_eval_strategy_steps_but_no_nsteps_provided(self):
        manager = TrainPipelineManager(objs_pool.assistant, objs_pool.trainer,
                                       objs_pool.validator, objs_pool.weights_saver,
                                       eval_strategy="steps")
        with self.assertRaises(ValueError):
            manager.run()
