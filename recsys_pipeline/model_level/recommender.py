import pandas as pd
import numpy as np


class Recommender:
    def __init__(self, user_items_loader_builder):
        self.loader_builder = user_items_loader_builder

    def score_all_items(self, user, assistant):
        all_items = assistant.get_all_items()
        dataloader = self.loader_builder.build(user, all_items)
        items_scores = self._pred_all_batches_flatten_result(dataloader, assistant)
        return items_scores

    def get_recommends(self, users, assistant):
        recommends = []
        for user in users:
            items_scores = self.score_all_items(user, assistant)
            recommended_items_idxs = np.argsort(items_scores)[::-1]
            user_recs = assistant.reverse_convert_items(recommended_items_idxs)
            recommends.append(user_recs)
        return recommends

    def _pred_all_batches_flatten_result(self, loader, assistant):
        all_preds = []
        for batch in loader:

            preds = assistant.preproc_then_forward(batch).squeeze(1).detach().numpy()
            all_preds += list(preds)
        return all_preds
