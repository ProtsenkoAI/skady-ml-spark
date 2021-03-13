from .data_managers.fit_data_managers import FitDataManager


class FitDataPreprocessor:
    def __init__(self, data_manager: FitDataManager):
        """
        :param data_manager: technology-specific (spark, local-machine fit, distributed etc) object
            with interface to process data
        """
        self.data_manager = data_manager

    def create_fit_data(self, new_data):
        unioned = self._accumulate_new_data(new_data)
        fit_data_raw = self._get_new_fit_data(unioned)
        fit_data = self.data_manager.prepare_for_fit(fit_data_raw)
        return fit_data

    def _accumulate_new_data(self, new_data):
        unioned_data = self.data_manager.union_with_saved_unused(new_data)
        return unioned_data

    def _get_new_fit_data(self, data):
        fit_data, unused_data = self.data_manager.split_data_to_fit_and_unused(data)
        self.data_manager.save_unused(unused_data)
        return fit_data
