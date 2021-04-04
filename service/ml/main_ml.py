# TODO: apply FAIR scheduler from spark (and test it)
# TODO: add expose/ to each package
from objs_creator import ObjsCreator
from model.expose.types import User
from typing import Optional


class ML:
    def __init__(self, config: dict):
        self.objs_creator = ObjsCreator(config)

        self.stream_fitter = self.objs_creator.get_stream_fitter()
        self.users_manager = self.objs_creator.get_users_manager()
        self.recommender = self.objs_creator.get_recommender()

    def start_fitting(self):
        data_obtainer = self.objs_creator.get_obtainer()
        fit_data = data_obtainer.get_data()
        self.stream_fitter.fit(fit_data)

    def await_fit(self, timeout: Optional[int] = None):
        self.stream_fitter.await_fit(timeout)

    def stop_fitting(self):
        self.stream_fitter.stop_fit()

    def get_recommends(self, user: User):
        recommends = self.recommender.get_recommends(user)
        return recommends

    def add_user(self, user: User):
        self.users_manager.add_user(user)

    def delete_user(self, user: User):
        self.users_manager.delete_user(user)
