from .base_processor import BaseProcessor


class DataProcessor(BaseProcessor):
    def preprocess_features(self, features):
        users, items = self.split_features(features, features_concated=True)
        users_tensor, items_tensor = self.preproc_users_items(users, items)
        return users_tensor, items_tensor
