from .base_processor import BaseProcessor
from pyspark.sql import DataFrame


class SparkProcessor(BaseProcessor):
    def preprocess_features(self, features: DataFrame):
        pd_features = features.toPandas()
        users, items = self.split_features(pd_features, features_concated=True)
        users_tensor, items_tensor = self.preproc_users_items(users, items)
        return users_tensor, items_tensor
