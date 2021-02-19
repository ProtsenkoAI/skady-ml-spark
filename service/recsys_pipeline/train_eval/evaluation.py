from sklearn import metrics
import numpy as np
import warnings


class Validator:
    def __init__(self, recommender, metric="ndcg"):
        self.recommender = recommender

        self.user_colname = "user_id"
        self.item_colname = "anime_id"
        self.label_colname = "rating"
        self.metric = metric

    def evaluate(self, assistant, interacts):
        scores = []
        for user in self._get_users(interacts):
            preds = self.recommender.score_all_items(user, assistant)
            items_true_scores = self._create_true_labels_for_every_item(user, interacts, assistant)
            user_score = self._score_preds(preds, items_true_scores)
            if not user_score is None:
                scores.append(user_score)
        if len(scores) == 0:
            return None
        return sum(scores) / len(scores)

    def _create_true_labels_for_every_item(self, user, interacts, assistant):
        all_items = assistant.get_all_items()
        # print("all_items", all_items)
        user_interacts = interacts[interacts[self.user_colname] == user]  # TODO: too low level operation for validator

        user_items = user_interacts[self.item_colname]
        user_labels = user_interacts[self.label_colname]
        # true_labels = dict(zip(user_items.values, user_labels.values))
        labels = {item: 0 for item in all_items}
        for item_id, item_rating in zip(user_items, user_labels):
            labels[item_id] = item_rating
        return labels

    def _get_users(self, interacts):
        return interacts[self.user_colname].unique()

    def _score_preds(self, preds, labels):
        # TODO: delete check of keys are equal later
        assert sum([not pred_key in labels for pred_key in preds.keys()]) == 0 #check that keys are equal
        some_keys = preds.keys()
        preds_proc = [preds[key] for key in some_keys]
        labels_proc = [labels[key] for key in some_keys]

        if self.metric == "ndcg":
            if len(labels) <= 1:
                warnings.warn("User has only one item for validation, so can't calc ndcg score, return None")
                val = None
            else:
                val = metrics.ndcg_score([labels_proc], [preds_proc])
        elif self.metric == "rmse":
            val = (metrics.mean_squared_error(preds_proc, labels_proc)) ** 0.5
        else:
            raise ValueError("metric name is wrong")
        return val

