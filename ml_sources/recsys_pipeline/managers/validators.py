import numpy as np
import pandas as pd
from sklearn import metrics


class Validator:
    def __init__(self, model, users_datasets_retriever, all_item_ids, preprocessor,
                 main_metric_name="rmse"):
        self.model = model
        self.users_datasets_retriever = users_datasets_retriever
        self.all_item_ids = all_item_ids
        self.main_metric_name = main_metric_name
        self.preprocessor = preprocessor

        self.labels_colname = "rating"
        self.items_colname = "anime_id"

    def evaluate(self):
        eval_vals = []
        for user, user_data in self.users_datasets_retriever: # iterating on batches
            user_eval_val = self._eval_user(user, user_data)
            eval_vals.append(user_eval_val)
        
        mean_metrics = pd.concat(eval_vals).mean()
        return self._get_main_metric_from_score(mean_metrics)

    def _eval_user(self, user, user_data) -> pd.DataFrame:
        user = self.preprocessor.preprocess_users(user)
        items_data = self.preprocessor.preprocess_items(self.all_item_ids)

        all_items_preds = self.model(user, items_data).squeeze().detach().numpy()

        items_preds_series = pd.Series(all_items_preds, index=self.all_item_ids)
        true_labels = self._label_all_items_from_interacts(user_data)
        score = self._score_preds(items_preds_series, true_labels)
        return score

    def _label_all_items_from_interacts(self, interacts):
        labels = interacts[self.labels_colname]
        items = interacts[self.items_colname]

        nitems = len(self.all_item_ids)
        item_labels = np.zeros(nitems)
        item_labels[items] = labels
        return item_labels

    def _score_preds(self, preds, true_labels):
        ndcg = metrics.ndcg_score([true_labels], [preds])
        rmse = (metrics.mean_squared_error(preds, true_labels)) ** 0.5
        return pd.DataFrame({"nDCG": ndcg, "rmse": rmse}, index=[0])

    def _get_main_metric_from_score(self, score: pd.DataFrame):
        main_metric = score[self.main_metric_name]
        # if metric value is better when it's smaller, we reverse it
        if self.main_metric_name in ["rmse"]:
            main_metric *= -1
        return main_metric
