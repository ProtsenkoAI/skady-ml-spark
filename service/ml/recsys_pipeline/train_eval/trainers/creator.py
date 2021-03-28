import os

from saving.savers import SparkSaver
from .spark_streaming_trainer import SparkStreamingTrainer
from .simple_trainer import SimpleTrainer


class TrainerCreator:
    def __init__(self, params, paths_info, mode):
        self.mode = mode
        self.params = params

        if self.mode == "spark":
            self.worker_dir = os.path.join(paths_info["base_path"],
                                           paths_info["worker_dir"])

        else:
            raise ValueError(self.mode)

    def get(self):
        # TODO: add typings and Trainer interface
        if self.mode == "spark":
            # TODO: replace torch' DataLoader with custom data iterator implementation (maybe?)
            fitter = SparkStreamingTrainer(self.params, self.worker_dir)
        elif self.mode == "local":
            fitter = SimpleTrainer()
        else:
            raise ValueError(f"mode is incorrect: {self.mode}")
        return fitter
