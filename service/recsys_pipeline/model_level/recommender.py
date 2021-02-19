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
            recommended_items_tuples = sorted(items_scores.items(), key=lambda item: -item[1])
            recommended_items = [item_and_label[0] for item_and_label in recommended_items_tuples]
            # user_recs = assistant.reverse_convert_items(recommended_items_idxs)
            recommends.append(recommended_items)
        return recommends

    def _pred_all_batches_flatten_result(self, loader, assistant):
        all_preds = {}
        for batch in loader:
            users, items = batch.split(1, dim=-1)  # TODO: too low level
            preds = assistant.preproc_forward_postproc(batch)
            item_to_score = dict(zip(items.detach().squeeze(-1).numpy(), preds))
            all_preds.update(item_to_score)
        return all_preds