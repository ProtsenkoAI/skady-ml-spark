from random import randint
import pandas as pd


class Validator:
    def __init__(self, recommender):
        self.recommender = recommender

        self.user_colname = "user_id"

    def evaluate(self, assistant, interacts):
        scores = []
        for user in self._get_users(interacts):
            preds = self.recommender.score_all_items(user, assistant)
            all_items_true_scores = self._get_user_item_scores(interacts, user, assistant.get_all_items())
            user_score = self.score_preds(preds, all_items_true_scores)
            scores.append(user_score)
        return sum(scores) / len(scores)

    def _get_users(self, interacts):
        return interacts[self.user_colname].unique()
