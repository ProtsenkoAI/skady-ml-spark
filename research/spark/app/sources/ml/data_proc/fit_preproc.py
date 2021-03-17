from .data_managers.fit_data_managers import FitDataManager


class FitDataPreprocessor:
    def __init__(self, data_manager: FitDataManager):
        """
        :param data_manager: technology-specific (spark, local-machine fit, distributed etc) object
            with interface to process data
        """
        self.data_manager = data_manager

    def create_fit_data(self, new_data_raw):
        fit_data = self.data_manager.prepare_for_fit(new_data_raw)
        return fit_data