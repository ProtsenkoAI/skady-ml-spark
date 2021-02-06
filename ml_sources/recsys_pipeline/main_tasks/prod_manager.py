import pandas as pd


class ProdManager:
    def __init__(self, trainer, recommender, saver, model_name, try_to_load=False, assistant_builder=None):
        self.trainer = trainer
        self.recommender = recommender
        self.saver = saver
        if try_to_load and self.saver.check_model_exists(model_name):
            self.assistant = self.saver.load(model_name)
        else:
            self.assistant = assistant_builder.build()

        self.interacts = None

    def add_interacts(self, new_interacts):
        new_inters_conved = self.assistant.update_and_convert(new_interacts)
        self._update_interacts(new_inters_conved)

    def _update_interacts(self, new_inters):
        if hasattr(self, "interacts"):
            merged_interacts = [self.interacts, new_inters]
        else:
            merged_interacts = [new_inters]
        self.interacts = pd.concat(merged_interacts, ignore_index=True)
