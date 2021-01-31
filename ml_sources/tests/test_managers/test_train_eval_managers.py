# TODO: add assertions using eval_res and saved_model

import unittest
import pandas as pd
import torch

from recsys_pipeline.models import mf_with_bias
from recsys_pipeline.managers import train_eval_managers
from recsys_pipeline.saving import meta_model_saving, model_state_dict_saving
from ..helpers import tests_config, objects_creation
config = tests_config.TestsConfig()


class TestTrainEvalManager(unittest.TestCase):
    def test_train_epochs_break_because_of_steps(self):
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        manager = train_eval_managers.TrainEvalManager(trainer, validator,
                                                       max_steps=10, max_epochs=5,
                                                       eval_strategy="epoch")

        eval_res = manager.train_eval()

    def test_train_steps_break_because_of_epochs(self):
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        manager = train_eval_managers.TrainEvalManager(trainer, validator,
                                                       max_steps=5000, max_epochs=1,
                                                       eval_strategy="steps",
                                                       nsteps=1001)
        eval_res = manager.train_eval()

    def test_train_steps_break_because_of_steps(self):
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        manager = train_eval_managers.TrainEvalManager(trainer, validator,
                                                       max_steps=10,
                                                       eval_strategy="steps",
                                                       nsteps=995)
        eval_res = manager.train_eval() # TODO: check eval_res is ok

    def test_train_epochs_break_because_of_epochs(self):
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        manager = train_eval_managers.TrainEvalManager(trainer, validator,
                                                       max_epochs=4,
                                                       eval_strategy="epoch")
        eval_res = manager.train_eval() # TODO: assert eval_res

    def test_exceed_stopping_patience(self):
        """We can't manually enforce eval value to decrease (to induce stopping because of the stopping patience
        exceeding. So we'll just set stopping patience, as well as max_steps and'll check that the manager
        will not break at functions devoted for stopping_patience checking
        """
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        manager = train_eval_managers.TrainEvalManager(trainer, validator,
                                                       eval_strategy="epoch",
                                                       stopping_patience=2,
                                                       max_steps=100)
        eval_res = manager.train_eval() # TODO

    def test_train_save_model(self):
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        model_saver = model_state_dict_saving.ModelStateDictSaver(save_dir=config.save_dir)

        manager = train_eval_managers.TrainEvalManager(trainer,
                                                       validator,
                                                       eval_strategy="steps",
                                                       save_best_model=True,
                                                       nsteps=1000,
                                                       max_steps=10,
                                                       model_saver=model_saver)
        print("before train_eval with saving")
        eval_results, best_model = manager.train_eval()
        print("res", eval_results, best_model)

    def test_raised_if_eval_strategy_steps_but_no_nsteps_specified(self):
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        model_saver = objects_creation.get_meta_saver()

        with self.assertRaises(ValueError):
            manager = train_eval_managers.TrainEvalManager(trainer,
                                                        validator,
                                                        eval_strategy="steps",
                                                        save_best_model=True,
                                                        #nsteps=1000 #nsteps not specified!
                                                        max_steps=3000,
                                                        model_saver=model_saver)
