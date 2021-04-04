from model import UsersManagerCreator, RecommenderCreator, UsersManager, Recommender
from train.expose import StreamingTrainer
from train.creators import TrainerCreator
from data import ObtainerCreator, Obtainer


class ObjsCreator:
    # TODO: do something with absolute paths in params
    # TODO: add saver creator
    def __init__(self, config):
        common_params = config["common_params"]

        self.obtainer_creator = ObtainerCreator(config, common_params)
        self.fitter_creator = TrainerCreator(config, common_params)
        self.users_manager = UsersManagerCreator(config, common_params)
        self.recommender_creator = RecommenderCreator(config, common_params)

    def get_users_manager(self) -> UsersManager:
        # TODO: creator for users manager
        return self.users_manager.get()

    def get_stream_fitter(self) -> StreamingTrainer:
        return self.fitter_creator.get_streaming()

    def get_obtainer(self) -> Obtainer:
        return self.obtainer_creator.get_fit_obtainer()

    def get_recommender(self) -> Recommender:
        return self.recommender_creator.get()
