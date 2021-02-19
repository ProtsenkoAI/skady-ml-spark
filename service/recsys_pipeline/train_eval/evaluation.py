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
            all_items = self._get_all_items(interacts)
            preds = self.recommender.score_all_items(user, assistant, all_items)
            items_true_scores = self._true_items_labels(user, interacts, all_items)
            user_score = self._score_preds(preds, items_true_scores)
            if not user_score is None:
                scores.append(user_score)
        return np.mean(scores)

    def _get_all_items(self, interacts):
        return interacts[self.item_colname].unique()

    def _true_items_labels(self, user, interacts, items):
        user_mask = interacts[self.user_colname] == user
        user_inters = interacts[user_mask]
        labels = []
        for item in items:
            user_item_inter = user_inters[user_inters[self.item_colname] == item]
            if len(user_item_inter) > 0:
                label = int(user_item_inter[self.label_colname])
            else:
                # user didn't interact with item
                label = 0
            labels.append(label)
        return labels

    def _get_users(self, interacts):
        return interacts[self.user_colname].unique()

    def _score_preds(self, preds, labels):
        if self.metric == "ndcg":
            if len(labels) <= 1:
                warnings.warn("User has only one item for validation, so can't calc ndcg score, return None")
                val = None
            else:
                val = metrics.ndcg_score([labels], [preds])
        elif self.metric == "rmse":
            val = (metrics.mean_squared_error(preds, labels)) ** 0.5
        else:
            raise ValueError("metric name is wrong")
        return val

