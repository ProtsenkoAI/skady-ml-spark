# TODO: apply Robert Martin's "layered architecture" from his blog
from recsys_pipeline.factory import ObjsCreator


class ML:
    # TODO: add typings and documentation
    def __init__(self, config):
        self.objs_creator = ObjsCreator(config)

        self.fitter = self.objs_creator.get_fitter()
        self.model = self.objs_creator.get_model_manager()
        self.recommender = self.objs_creator.get_recommender()

    def start_fitting(self):
        # TODO: ensure that fitting isn't terminated when start_fitting() function ends
        # TODO: add stop functionality to start_fitting() (maybe in threading-like fashion)
        data_obtainer = self.objs_creator.get_data_obtainer()
        fit_data = data_obtainer.get_fit_data()
        self.fitter.fit(self.model, fit_data)

    def stop_fitting(self):
        self.fitter.stop_fit()

    def get_recommends(self, user):
        # return self.model.predict_user(user)
        all_items = self.model.processor.get_all_items()
        print("in recommends", user, all_items)
        recommends = self.recommender.get_recommends([user], self.model, all_items)
        return recommends

    def add_user(self, user):
        # TODO: add some user abstraction
        self.model.add_user(user)

    def delete_user(self, user):
        self.model.delete_user(user)
