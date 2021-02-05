from collections.abc import Iterable

from data_transform import id_idx_conv, preprocessing


class ModelAssistant:
    def __init__(self):
        self.user_conv = id_idx_conv.IdIdxConverter()
        self.item_conv = id_idx_conv.IdIdxConverter()
        self.tensor_creator = preprocessing.TensorCreator(device="cpu")

        self.user_colname = "user_id"
        self.item_colname = "anime_id"

    def get_all_items(self):
        return self.item_conv.get_all_ids()

    def preproc_then_forward(self, model, features, parts_concated=True):
        proc_features = self.preprocess_features(features, parts_concated)
        preds = model.forward(*proc_features)
        return preds

    def update_with_new_interacts(self, model, interacts):
        users = interacts[self.user_colname]
        items = interacts[self.item_colname]
        user_indexes = self.user_conv.add_ids_get_idxs(*users)
        item_indexes = self.item_conv.add_ids_get_idxs(*items)

        users_needed_for_interacts = max(user_indexes) + 1
        items_needed_for_interacts = max(item_indexes) + 1
        self._enlarge_model_if_needed(model, users_needed_for_interacts, items_needed_for_interacts)

    def _enlarge_model_if_needed(self, model, need_users, need_items):
        model_stats = model.get_init_kwargs()
        curr_nusers = model_stats["nusers"]
        curr_nitems = model_stats["nitems"]

        diff_users = need_users - curr_nusers
        if diff_users > 0:
            model.add_users(diff_users)
        diff_items = need_items - curr_nitems
        if diff_items > 0:
            model.add_items(diff_items)

    def preprocess_features(self, features, parts_concated=False):
        users, items = self._split_users_items(features, parts_concated)
        if not isinstance(users, Iterable):
            users = (users,)
        users_tensor = self.preprocess_users(*users)
        items_tensor = self.preprocess_items(*items)
        return users_tensor, items_tensor

    def preproc_labels(self, labels):
        return self.tensor_creator.get_labels_tensor(labels)

    def preprocess_users(self, *users):
        users_idxs = self.user_conv.get_idxs(*users)
        users_tensor = self.tensor_creator.get_users_tensor(users_idxs)
        return users_tensor

    def preprocess_items(self, *items):
        items_idxs = self.item_conv.get_idxs(*items)
        items_tensor = self.tensor_creator.get_items_tensor(items_idxs)
        return items_tensor

    def postprocess_recommends(self, users_preds):
        processed = []
        for user_recommendations in users_preds:
            user_proc = self.item_conv.get_idxs(*user_recommendations)
            processed.append(user_proc)
        return processed

    def _split_users_items(self, features, parts_concated):
        if parts_concated:
            users, items = zip(*features)
        else:
            users, items = features
        return users, items
