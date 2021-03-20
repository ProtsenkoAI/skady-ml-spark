# TODO: move recsys_pipeline and research/spark/app to ml/
from recsys_pipeline.factory import ObjsCreator


class ML:
    # TODO: add typings and documentation
    # TODO: ensure can use asynchronously
    def __init__(self, config):
        self.objs_creator = ObjsCreator(config)

        self.fitter = self.objs_creator.get_fitter()
        self.data_obtainer = self.objs_creator.get_data_obtainer()

    def start_fitting(self):
        fit_data = self.data_obtainer.get_fit_data()
        model = self.objs_creator.get_model()
        self.fitter.fit(model, fit_data)

    def stop_fitting(self):
        self.fitter.stop_fit()

    def get_recommends(self, user):
        model = self.objs_creator.get_model()
        return model.predict_user(user)

    def add_user(self, user):
        model = self.objs_creator.get_model()
        model.add_user(user)

    def delete_user(self, user):
        model = self.objs_creator.get_model()
        model.delete_user(user)
