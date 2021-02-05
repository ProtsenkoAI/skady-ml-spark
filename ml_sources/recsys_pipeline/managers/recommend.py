import numpy as np

from . import assistants


class Recommender:
    def __init__(self, assistant: assistants.ModelAssistant, batch_size=64):
        self.assistant = assistant
        self.batch_size = batch_size

    def get_recommends(self, model, users_ids):
        items_probs = self.get_items_probs(model, users_ids)
        sorted_indexes = np.argsort(items_probs, axis=1)[::-1] # inverting: from highest rating to lowest
        sorted_ids = self.assistant.postprocess_recommends(sorted_indexes)
        return sorted_ids

    def get_items_probs(self, model, user_ids):
        all_items = self.assistant.get_all_items()
        item_batches = self._create_batches(all_items, self.batch_size)

        users_probas = []
        for user in user_ids:
            users_probas.append([])
            for batch in item_batches:
                preds = self.assistant.preproc_then_forward(model, (user, batch), parts_concated=False)
                users_probas[-1] += list(preds)
        users_probas = np.array(users_probas)
        return users_probas

    def _create_batches(self, values, batch_size):
        batches = [values[start_idx: start_idx + batch_size]
                   for start_idx in range(0, len(values), batch_size)]
        return batches
