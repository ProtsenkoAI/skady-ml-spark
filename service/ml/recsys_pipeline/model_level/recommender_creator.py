from .recommender import Recommender
from data.building_loaders import UserItemsLoaderBuilder


class RecommenderCreator:
    def __init__(self, params):
        self.batch_size = params["batch_size"]

    def get(self):
        load_builder = UserItemsLoaderBuilder(batch_size=self.batch_size)
        return Recommender(load_builder)
