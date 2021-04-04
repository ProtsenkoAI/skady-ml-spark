from ..expose.model_manager import ModelManager
from ..expose.recommender import Recommender
from typing import List


class LocalRecommender(Recommender):
    User = int

    def __init__(self, user_items_loader_builder, model_manager: ModelManager):
        self.loader_builder = user_items_loader_builder
        self.model_manager = model_manager

    def get_recommends(self, user: User):
        all_items = self.model_manager.get_all_users()
        filtered_items = self.filter_items(all_items)
        if len(filtered_items) == 0:
            raise ValueError(f"There are no any items left for user {user}")
        items_scores = self.score_all_items(user, self.model_manager, filtered_items)
        sorted_recommends = self.sort_items_by_scores(filtered_items, items_scores)
        return sorted_recommends

    def filter_items(self, all_items: List[int]):
        # TODO: filter with business rules
        return all_items

    def sort_items_by_scores(self, items, scores):
        recommended_items = [item for score, item in sorted(zip(scores, items),
                                                            key=lambda pair: -pair[0])]
        return recommended_items

    def score_all_items(self, user, assistant, all_items):
        dataloader = self.loader_builder.build(user, all_items)
        items_scores = self._pred_all_batches_flatten_result(dataloader, assistant)
        return items_scores

    @staticmethod
    def _pred_all_batches_flatten_result(loader, assistant):
        all_preds = []
        for batch in loader:
            preds = assistant.preproc_forward_postproc(batch)
            all_preds += preds
        return all_preds
