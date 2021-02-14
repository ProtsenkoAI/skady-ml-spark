import pandas as pd


class ProdManager:
    def __init__(self, trainer, recommender, saver, model_name, try_to_load=True, assistant_builder=None,
                 dataloader_builder=None):
        self.trainer = trainer
        self.recommender = recommender
        self.saver = saver
        self.dataloader_builder = dataloader_builder
        if try_to_load and self.saver.check_model_exists(model_name):
            self.assistant = self.saver.load(model_name)
        else:
            self.assistant = assistant_builder.build()

        self.interacts = None
        self.saved_model_name = None

    def add_interacts(self, new_interacts):
        new_interacts = new_interacts.copy()
        self.assistant.update_with_interacts(new_interacts)
        self._update_interacts(new_interacts)

    def fit(self, nepochs=None, nsteps=None):
        # TODO: hide all low-level things inside trainer and assistant, not prod_manager!
        dataloader = self.dataloader_builder.build(self.interacts)
        self.trainer.fit(self.assistant, dataloader, nsteps=nsteps, nepochs=nepochs)

    def get_recommends(self, users):
        recommends = self.recommender.get_recommends(users, self.assistant)
        return recommends

    def save(self):
        self.saved_model_name = self.saver.save(self.assistant)

    def load(self):
        self.assistant = self.saver.load(self.saved_model_name)

    def _update_interacts(self, new_inters):
        if hasattr(self, "interacts"):
            merged_interacts = [self.interacts, new_inters]
        else:
            merged_interacts = [new_inters]
        self.interacts = pd.concat(merged_interacts, ignore_index=True)


