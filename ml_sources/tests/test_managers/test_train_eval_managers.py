import unittest
import pandas as pd
import torch

from recsys_pipeline.models import mf_with_bias
from recsys_pipeline.managers import train_eval_managers
from recsys_pipeline.saving import meta_model_saving, model_state_dict_saving
from ..helpers import tests_config, objects_creation
from . import helpers
config = tests_config.TestsConfig()

class TestTrainEvalManager(unittest.TestCase):
    def test_train_epochs_break_because_of_steps(self):
        # trainer = TrainerTestingPlug(4000)
        # validator = ValidatorTestingPlug([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        manager = train_eval_managers.TrainEvalManager(trainer, validator, 
                                                       max_steps=5000, max_epochs=5,
                                                       eval_strategy="epoch")

        eval_res = manager.train_eval()
        eval_rounds_passed = validator.eval_idx
        self.assertEqual(eval_rounds_passed, 2)

    def test_train_steps_break_because_of_epochs(self):
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        manager = train_eval_managers.TrainEvalManager(trainer, validator, 
                                                       max_steps=5000, max_epochs=1,
                                                       eval_strategy="steps",
                                                       nsteps=1001)
        eval_res = manager.train_eval()
        eval_rounds_passed = validator.eval_idx
        self.assertEqual(eval_rounds_passed, 4)

        nsteps_by_train_cycle = trainer.saved_nsteps
        self.assertEqual(nsteps_by_train_cycle, [1001, 1001, 1001, 997])
        
    def test_train_steps_break_because_of_steps(self):
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        manager = train_eval_managers.TrainEvalManager(trainer, validator, 
                                                       max_steps=3000, 
                                                       eval_strategy="steps",
                                                       nsteps=995)
        manager.train_eval()

        rounds_passed = validator.eval_idx
        self.assertEqual(rounds_passed, 4)

        nsteps_by_train_cycle = trainer.saved_nsteps
        self.assertEqual(nsteps_by_train_cycle, [995, 995, 995, 15])


    def test_train_epochs_break_because_of_epochs(self):
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        manager = train_eval_managers.TrainEvalManager(trainer, validator, 
                                                       max_epochs=4,
                                                       eval_strategy="epoch")
        manager.train_eval()
        # rounds_passed = validator.eval_idx
        # self.assertEqual(rounds_passed, 4)

        nsteps_by_train_cycle = trainer.saved_nsteps
        self.assertEqual(nsteps_by_train_cycle, [4000, 4000, 4000, 4000])

    def test_exceed_stopping_patience(self):
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        manager = train_eval_managers.TrainEvalManager(trainer, validator, 
                                                       eval_strategy="epoch",
                                                       stopping_patience=2)
        manager.train_eval()

        # rounds_passed = validator.eval_idx
        # self.assertEqual(rounds_passed, 4)

    def test_train_save_model(self):
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        model_saver = model_state_dict_saving.ModelStateDictSaver()

        manager = train_eval_managers.TrainEvalManager(trainer, 
                                                       validator, 
                                                       eval_strategy="steps",
                                                       save_best_model=True,
                                                       nsteps=1000,
                                                       max_steps=3000,
                                                       model_saver=model_saver)

        eval_results, best_model = manager.train_eval()
        # self.assertEqual(eval_results, [0.1, 0.2, 0.1])

    def test_raised_if_eval_strategy_steps_but_no_nsteps_specified(self):
        # trainer = TrainerTestingPlug(4000)
        # validator = ValidatorTestingPlug([0.1, 0.2, 0.1, 0.3, 0.2, 0.2])
        # model_saver = model_state_dict_saving.ModelStateDictSaver()
        model = objects_creation.get_mf_model(20, 20)
        trainer = objects_creation.get_trainer(model)
        validator = objects_creation.get_validator(model)

        model_saver = object_creation.get_meta_saver()

        with self.assertRaises(ValueError):
            manager = train_eval_managers.TrainEvalManager(trainer, 
                                                        validator, 
                                                        eval_strategy="steps",
                                                        save_best_model=True,
                                                        #nsteps=1000 #nsteps not specified!
                                                        max_steps=3000,
                                                        model_saver=model_saver)
