from ..model_level import ManagerCreator, RecommenderCreator
from ..train_eval import TrainerCreator
from ..data.obtain import ObtainerCreator


class ObjsCreator:
    # TODO: do something with absolute paths in params
    def __init__(self, config):
        mode = config["mode"]
        manager_params = config["model_manager_params"]
        obtainer_params = config["obtainer_params"]
        fitter_params = config["fitter_params"]
        recommender_params = config["recommender_params"]
        paths_info = config["paths"]

        self.manager_creator = ManagerCreator(manager_params)
        self.obtainer_creator = ObtainerCreator(obtainer_params, paths_info, mode)
        self.fitter_creator = TrainerCreator(fitter_params, paths_info, mode)
        self.recommender_creator = RecommenderCreator(recommender_params)

    def get_model_manager(self):
        return self.manager_creator.get()

    def get_fitter(self):
        return self.fitter_creator.get()

    def get_obtainer(self):
        return self.obtainer_creator.get()

    def get_recommender(self):
        return self.recommender_creator.get()
