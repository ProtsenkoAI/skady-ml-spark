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
            all_items_true_scores = self._create_true_labels_for_every_item(user, interacts, assistant)
            # print("preds", preds)
            user_score = self._score_preds(preds, all_items_true_scores)
            if not user_score is None:
                scores.append(user_score)
        if len(scores) == 0:
            return None
        return sum(scores) / len(scores)

    def _create_true_labels_for_every_item(self, user, interacts, assistant):
        all_items = assistant.get_all_items()
        # print("all_items", all_items)
        user_interacts = interacts[interacts[self.user_colname] == user]
        # convert item ids to indexes
        item_idxs = assistant.convert_items(user_interacts[self.item_colname])
        labels = interacts[self.label_colname]

        true_labels = np.zeros(len(all_items))
        true_labels[item_idxs] = labels
        return true_labels

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

