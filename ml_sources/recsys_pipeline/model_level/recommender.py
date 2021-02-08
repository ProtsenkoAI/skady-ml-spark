import pandas as pd


class Recommender:
    def __init__(self, loader_builder_from_users_and_items):
        self.loader_builder_from_users_and_items = loader_builder_from_users_and_items

    def recommend(self, assistant, users):
        all_items = assistant.get_all_items()
        loader = self.loader_builder_from_users_and_items.build(users, all_items)
        preds = self._get_users_preds(assistant, loader)

    def _get_users_preds(self, assistant, loader):
        users_items_preds = pd.DataFrame(columns=["users", "items", "preds"])
        for batch in loader:
            preds = assistant.predict(batch)
            batch_tmp_df = pd.DataFrame({"users": users, "items": items, "preds": preds})
            users_items_preds = pd.concat([users_items_preds, batch_tmp_df], ignore_index=True)
        users_groups = [group.sort_values("items")["preds"] for (_, group) in users_items_preds.groupby("users")]
        #print("users_groups", users_groups)