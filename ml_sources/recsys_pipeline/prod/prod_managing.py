import pandas as pd

from recsys_pipeline.data_transform import id_idx_conv, preprocessing
from recsys_pipeline.managers import trainers


class ProdManager:
    def __init__(self,
                 model_saver,
                 preprocessor: preprocessing.TensorCreator,
                 dataloader_builder,
                 train_kwargs={},
                 model_init_kwargs={},
                 model_builder=None,
                 try_to_load_model=True):

        self.model_saver = model_saver

        self.user_colname = "user_id"
        self.item_colname = "anime_id"

        self.preprocessor = preprocessor
        self.dataloader_builder = dataloader_builder
        self.train_kwargs = train_kwargs
        self.model_init_kwargs = model_init_kwargs
        self.model_builder = model_builder

        self.try_to_load_model = try_to_load_model

        if self.try_to_load_model and self.model_saver.check_model_exists():
            load_res = self._load_model_and_converters()
            self.model, self.user_conv, self.item_conv = load_res
            # achtung! Using none because have no interacts, may cause bugs
            self.trainer = self._create_trainer(self.model, None)
        else:
            creation_res = self.create_model_and_converters()
            self.model, self.user_conv, self.item_conv = creation_res
            self.trainer = None

    def save(self):
        self.model_saver.save(self.model, self.user_conv, self.item_conv)

    def create_model_and_converters(self):
        model = self.model_builder(**self.model_init_kwargs)
        user_conv = id_idx_conv.IdIdxConverter()
        item_conv = id_idx_conv.IdIdxConverter()
        return model, user_conv, item_conv

    def add_interacts(self, new_interacts):
        new_inters_conved = self._add_to_convs_and_convert_interacts(new_interacts)
        self._update_interacts(new_inters_conved)
        self._add_new_users_and_items_to_model(new_inters_conved)

    def fit(self, nepochs=None, nsteps=None):
        print(self.interacts.describe())
        self.trainer = self._create_trainer(self.model, self.interacts)
        self.trainer.fit(nepochs, nsteps)

    def get_recommends(self, users):
        users_conved = self.user_conv.get_idxs(*users)
        users_processed = self.preprocessor.get_users_tensor(users_conved)
        all_item_idxs = self.item_conv.get_all_idxs()
        proc_item_idxs = self.preprocessor.get_items_tensor(all_item_idxs)
        all_items_preds = self.trainer.get_recommends_for_users(users_processed, proc_item_idxs)  # omg wtf
        user_preds = []
        for user in all_items_preds:
            user_conv = self.item_conv.get_ids(*user)
            user_preds.append(user_conv)
        return user_preds

    def _load_model_and_converters(self):
        model, (user_conv, item_conv) = self.model_saver.load()
        return model, user_conv, item_conv

    def _add_to_convs_and_convert_interacts(self, interacts):
        users = interacts[self.user_colname]
        items = interacts[self.item_colname]
        interacts[self.user_colname] = self.user_conv.add_ids_get_idxs(*users)
        interacts[self.item_colname] = self.item_conv.add_ids_get_idxs(*items)
        return interacts

    def _update_interacts(self, new_interacts_conved):
        if hasattr(self, "interacts"):
            merged_interacts = [self.interacts, new_interacts_conved]
        else:
            merged_interacts = [new_interacts_conved]

        self.interacts = pd.concat(merged_interacts, ignore_index=True)

    def _add_new_users_and_items_to_model(self, interacts):
        users = interacts[self.user_colname]
        items = interacts[self.item_colname]
        nusers_in_interacts = len(users.unique())
        nitems_in_interacts = len(items.unique())
        model_kwargs = self.model.get_init_kwargs()
        nusers_in_model = model_kwargs["nusers"]
        nitems_in_model = model_kwargs["nitems"]
        new_users = max(nusers_in_interacts - nusers_in_model, 0)
        new_items = max(nitems_in_interacts - nitems_in_model, 0)

        self.model.add_users(new_users)
        self.model.add_items(new_items)

    def _create_trainer(self, model, interacts):
        dataset = self.dataloader_builder(interacts)
        trainer = trainers.Trainer(model, dataset, self.preprocessor, **self.train_kwargs)
        return trainer
