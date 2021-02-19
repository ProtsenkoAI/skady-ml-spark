import pandas as pd


class RecommendManager:
    def __init__(self, trainer, recommender, saver, assistant, fit_epochs=2, fit_steps=None):
        self.fit_epochs = fit_epochs
        self.fit_steps = fit_steps

        self.trainer = trainer
        self.recommender = recommender
        self.saver = saver
        self.assistant = assistant

        self.interacts = None
        self.model_name = None
        self.item_colname = "anime_id"

    def fit(self, new_interacts):
        self._add_interacts(new_interacts)
        self.trainer.fit(self.assistant, self.interacts, max_step=self.fit_steps,
                         max_epoch=self.fit_epochs)
        self.model_name = self._save()

    def _add_interacts(self, new_interacts):
        new_interacts = new_interacts.copy()
        self.assistant.update_with_interacts(new_interacts)
        self._update_interacts(new_interacts)

    def get_recommends(self, users):
        all_items = self._get_all_items()
        recommends = self.recommender.get_recommends(users, self.assistant, all_items)
        return recommends

    def _get_all_items(self):
        return self.interacts[self.item_colname].unique()

    def _save(self):
        model_name = self.assistant.save(self.saver)
        return model_name

    def _update_interacts(self, new_inters):
        if not self.interacts is None:
            all_interacts = [self.interacts, new_inters]
        else:
            all_interacts = [new_inters]
        self.interacts = pd.concat(all_interacts, ignore_index=True)
