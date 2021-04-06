from ..expose import ManagerSaver, RecsysTorchModel, Processor
from typing import Tuple
from . import util


class SparkManagerSaver(ManagerSaver):
    def __init__(self, save_path: str):
        self.save_path = save_path

    def load(self) -> Tuple[RecsysTorchModel, Processor]:
        model, processor = util.load_dill(self.save_path)
        return model, processor

    def save(self, model: RecsysTorchModel, processor: Processor):
        util.dump_dill((model, processor), self.save_path)
