from .simple_obtainer import SimpleObtainer
from .spark_obtainer import SparkObtainer


class ObtainerCreator:
    def __init__(self, params: dict, paths: dict, mode: str):
        self.params = params
        self.paths_params = paths
        self.mode = mode

        if self.mode not in ["spark", "local"]:
            raise ValueError(self.mode)

    def get(self):
        if self.mode == "local":
            return SimpleObtainer(self.params)

        elif self.mode == "spark":
            return SparkObtainer(self.params, self.paths_params)
