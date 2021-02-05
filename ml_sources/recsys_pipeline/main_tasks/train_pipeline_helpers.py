import numpy as np


def steps_left_from_max_steps_epochs(max_steps, max_epochs, steps_in_epoch):
    if max_steps is None and max_epochs is None:
        return None
    if not max_epochs is None:
        max_steps_because_of_epochs = max_epochs * steps_in_epoch
        if not max_steps is None:
            max_steps = min(max_steps, max_steps_because_of_epochs)
        else:
            max_steps = max_steps_because_of_epochs
    return max_steps


def decide_continue_training(eval_results, stop_patience, steps_left):
    if not steps_left is None: # can be None if both max_steps and max_epochs are None
        if steps_left <= 0:
            return False

    if not stop_patience is None and len(eval_results) > 0:
        steps_passed = len(eval_results)
        best_result_step = np.argmax(eval_results)
        if (steps_passed - best_result_step - 1) >= stop_patience:
            print("EARLY STOPPING", "eval_vals", eval_results, "best_result_step", best_result_step)
            return False
    return True


def get_train_kwargs(nsteps, steps_left, eval_strategy):
    if eval_strategy == "epochs":
        train_kwargs = {"nepochs": 1}
    elif eval_strategy == "steps":
        if nsteps is None:
            raise ValueError("if eval_strategy is steps nsteps can't be None")
        if steps_left is None:
            nsteps = nsteps
        else:
            nsteps = min(nsteps, steps_left)
        train_kwargs = {"nsteps": nsteps}
    else:
        raise ValueError("incorrect eval_strategy_value")
    return train_kwargs


def update_steps_left(steps_left, train_kwargs, steps_in_epoch):
    if steps_left is None:
        return steps_left
    if "nepochs" in train_kwargs:
        nepochs = train_kwargs["nepochs"]
        if not nepochs is None:
            steps_left -= nepochs * steps_in_epoch
    elif "nsteps" in train_kwargs:
        nsteps = train_kwargs["nsteps"]
        if not nsteps is None:
            steps_left -= nsteps
    return steps_left


def has_to_save(eval_results, save_best):
    if not save_best:
        return False

    last_is_best = np.argmax(eval_results) == (len(eval_results) - 1)
    return last_is_best
