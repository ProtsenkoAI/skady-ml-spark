from .base_processor import BaseProcessor


class SparkProcessor(BaseProcessor):
    def preprocess_features(self, features):
        # TODO: if works, delete the processor.
        #   from the archit. side, incoming data format shouldn't touch model level
        pd_features = features
        users, items = self.split_features(pd_features, features_concated=False)
        users_tensor, items_tensor = self.preproc_users_items(users, items)
        return users_tensor, items_tensor
