from collections.abc import Iterable
import numpy as np

from data_transform import id_idx_conv, preprocessing
# from data import loader_factories


class ModelAssistant:
    def __init__(self, model, batch_size=64):
        self.model = model
        self.user_conv = id_idx_conv.IdIdxConverter()
        self.item_conv = id_idx_conv.IdIdxConverter()
        self.tensor_creator = preprocessing.TensorCreator(device="cpu")
        self.batch_size = batch_size

        self.user_colname = "user_id"
        self.item_colname = "anime_id"

    def preproc_labels(self, labels):
        return self.tensor_creator.get_labels_tensor(labels)

    def preproc_then_forward(self, features, parts_concated=True):
        proc_features = self.preprocess_features(features, parts_concated)
        preds = self.model.forward(*proc_features)
        return preds

    def get_recommends(self, users_ids):
        items_probas = self.get_probas_with_all_items(users_ids)
        sorted_indexes = np.argsort(items_probas, axis=1)[::-1] # inverting: from highest rating to lowest
        sorted_ids = self.postprocess_recommends(sorted_indexes)
        return sorted_ids

    def get_probas_with_all_items(self, user_ids):
        all_items = self.item_conv.get_all_ids()
        item_batches = self._create_batches(all_items, self.batch_size)

        users_probas = []
        for user in user_ids:
            users_probas.append([])
            for batch in item_batches:
                preds = self.preproc_then_forward((user, batch), parts_concated=False)
                users_probas[-1] += list(preds)
        users_probas = np.array(users_probas)
        return users_probas

    def _create_batches(self, values, batch_size):
        batches = [values[start_idx: start_idx + batch_size]
                   for start_idx in range(0, len(values), batch_size)]
        return batches

    def update_with_new_interacts(self, interacts):
        users = interacts[self.user_colname]
        items = interacts[self.item_colname]
        user_indexes = self.user_conv.add_ids_get_idxs(*users)
        item_indexes = self.item_conv.add_ids_get_idxs(*items)

        users_needed_for_interacts = max(user_indexes) + 1
        items_needed_for_interacts = max(item_indexes) + 1
        self._enlarge_model_if_needed(users_needed_for_interacts, items_needed_for_interacts)

    def _enlarge_model_if_needed(self, need_users, need_items):
        model_stats = self.model.get_init_kwargs()
        curr_nusers = model_stats["nusers"]
        curr_nitems = model_stats["nitems"]

        diff_users = need_users - curr_nusers
        if diff_users > 0:
            self.model.add_users(diff_users)
        diff_items = need_items - curr_nitems
        if diff_items > 0:
            self.model.add_items(diff_items)

    def preprocess_features(self, features, parts_concated=False):
        users, items = self._split_users_items(features, parts_concated)
        if not isinstance(users, Iterable):
            users = (users,)
        users_tensor = self.preprocess_users(*users)
        items_tensor = self.preprocess_items(*items)
        return users_tensor, items_tensor

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