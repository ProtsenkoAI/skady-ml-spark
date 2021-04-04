from ..expose import RecommendsPostprocessor
from ..recommends_processing import RecommendsPostprocessorImpl


class RecommendsPostprocessorCreator:
    def __init__(self, config: dict, common_params: dict):
        self.config = config
        self.common_params = common_params

    def get(self) -> RecommendsPostprocessor:
        return RecommendsPostprocessorImpl()
