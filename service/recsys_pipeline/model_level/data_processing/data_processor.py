class DataProcessor:
    def __init__(self, user_conv, item_conv, tensor_creator):
        self.user_conv = user_conv
        self.item_conv = item_conv
        self.tensor_creator = tensor_creator

        self.user_colname = "user_id"
        self.item_colname = "anime_id"

    def preprocess_features(self, features):
        users, items = self._split_features(features, features_concated=True)
        users_conved, items_conved = self.convert_users_items(users, items)
        users_tensor = self.tensor_creator.get_feature_tensor(users_conved)
        items_tensor = self.tensor_creator.get_feature_tensor(items_conved)
        return users_tensor, items_tensor

    def convert_users_items(self, users, items):
        proc_users = self.convert_users(*users)
        proc_items = self.convert_items(*items)
        return proc_users, proc_items

    def convert_users(self, *users):
        return self.user_conv.get_idxs(*users)

    def convert_items(self, *items):
        return self.item_conv.get_idxs(*items)

    def reverse_convert_items(self, *items):
        return self.item_conv.get_ids(*items)

    def preprocess_labels(self, labels):
        return self.tensor_creator.get_labels_tensor(labels)

    def _split_features(self, features, features_concated=True):
        if features_concated:
            users, items = zip(*features)
        else:
            users, items = features
        return users, items

    def get_convs_data(self):
        users_data = self.get_user_conv().dump()
        items_data = self.get_item_conv().dump()
        return users_data, items_data

    def get_user_conv(self):
        return self.user_conv

    def get_item_conv(self):
        return self.item_conv

    def get_nusers_nitems(self):
        users = self.user_conv.get_all_idxs()
        items = self.item_conv.get_all_idxs()
        return len(users), len(items)

    def update(self, interacts):
        interacts = interacts.copy()
        users, items = self._get_users_items(interacts)
        self.user_conv.add_ids(*users)
        self.item_conv.add_ids(*items)

    def postproc_preds(self, preds):
        return list(preds.squeeze(1).cpu().detach().numpy())

    def _convert_users(self, users):
        return self.user_conv.get_idxs(*users)

    def _convert_items(self, items):
        return self.item_conv.get_idxs(*items)

    def _get_users_items(self, interacts):
        users = interacts[self.user_colname]
        items = interacts[self.item_colname]
        return users, items
