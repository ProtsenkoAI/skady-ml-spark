import pandas as pd
import numpy as np


class Recommender:
    def __init__(self, user_items_loader_builder):
        self.loader_builder = user_items_loader_builder

    def score_all_items(self, user, assistant, all_items):
        dataloader = self.loader_builder.build(user, all_items)
        items_scores = self._pred_all_batches_flatten_result(dataloader, assistant)
        return items_scores

    def get_recommends(self, users, assistant, all_items):
        recommends = []
        for user in users:
            items_scores = self.score_all_items(user, assistant, all_items)
            # recommended_items = np.argsort(items_scores)[::-1]
            recommended_items = [item for score, item in sorted(zip(items_scores, all_items),
                                                                key=lambda pair: -pair[0])]
            recommends.append(recommended_items)
        return recommends

    def _pred_all_batches_flatten_result(self, loader, assistant):
        all_preds = []
        for batch in loader:
            preds = assistant.preproc_forward_postproc(batch)
            all_preds += preds
        return all_preds
