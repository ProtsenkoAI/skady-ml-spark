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

    # def test_eval_every_nsteps_break_because_of_steps(self):
    #     ...
    #
    # def test_eval_every_epoch_break_because_of_epochs(self):
    #     ...
    #
    # def test_eval_every_nsteps_break_because_of_epochs(self):
    #     ...
    #
    # def test_exceed_stopping_patience(self):
    #     ...
    #
    # def test_raises_error_if_eval_strategy_steps_but_no_nsteps_provided(self):
    #     ...
