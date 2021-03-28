# TODO: apply Robert Martin's "layered architecture" from his blog
from recsys_pipeline.factory import ObjsCreator
from ml_types import User


class ML:
    # TODO: add typings and documentation
    def __init__(self, config: dict):
        self.objs_creator = ObjsCreator(config)

        self.fitter = self.objs_creator.get_fitter()
        self.model = self.objs_creator.get_model_manager()
        self.recommender = self.objs_creator.get_recommender()

    def start_fitting(self):
        # TODO: ensure that fitting isn't terminated when start_fitting() function ends
        data_obtainer = self.objs_creator.get_obtainer()
        fit_data = data_obtainer.get_fit_data()
        self.fitter.fit(self.model, fit_data)

    def await_fit(self, timeout=None):
        self.fitter.await_fit(timeout)

    def stop_fitting(self):
        self.fitter.stop_fit()

    def get_recommends(self, user: User):
        all_items = self.model.get_all_users()
        # TODO: maybe add some sort of preprocessing?
        recommends = self.recommender.get_recommends([user], self.model, all_items)[0]
        return recommends

    def add_user(self, user: User):
        # TODO: add some user abstraction (not int)
        self.model.add_user(user)

    def delete_user(self, user: User):
        self.model.delete_user(user)
