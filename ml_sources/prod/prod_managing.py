import sys
import pandas as pd

sys.path.insert(0, '..')

from recsys_pipeline.data_transform import id_idx_converter, preprocessing
from recsys_pipeline.managers import trainers


class ProdManager:
    # TODO: write dataloader_builder
    def __init__(self,
                 model_name,
                 model_saver,
                 preprocessor: preprocessing.DataPreprocessor,
                 dataloader_builder,
                 train_kwargs={},
                 model_init_kwargs={},
                 model_builder=None,
                 try_to_load_model=True):

        self.model_name = model_name
        self.model_saver = model_saver

        self.user_colname = "user_id"
        self.item_colname = "anime_id"

        self.preprocessor = preprocessor
        self.dataloader_builder = dataloader_builder
        self.train_kwargs = train_kwargs
        self.model_init_kwargs = model_init_kwargs
        self.model_builder = model_builder

        self.try_to_load_model = try_to_load_model

        if self.try_to_load_model and self.model_saver.check_model_exists(self.model_name):
            load_res = self.load_model_and_converters()
            self.model, self.user_conv, self.item_conv = load_res
        else:
            creation_res = self.create_model_and_converters()
            self.model, self.user_conv, self.item_conv = creation_res

        # self.trainer = self._create_trainer(self.model)

    def save(self):
        users_ids = self.user_conv.get_all_ids()
        items_ids = self.item_conv.get_all_ids()
        self.model_saver.save(self.model_name, self.model.state_dict(), users_ids, items_ids, meta_info=self.model_init_kwargs)

    def create_model_and_converters(self):
        model = self.model_builder(**self.model_init_kwargs)
        user_conv = id_idx_converter.IdIdxConverter()
        item_conv = id_idx_converter.IdIdxConverter()
        return model, user_conv, item_conv

    def add_interacts(self, new_interacts):
        conv_new_interacts = self.convert_interacts_add_to_convs(new_interacts)
        if hasattr(self, "interacts"):
            merged_interacts = [self.interacts, conv_new_interacts]
        else:
            merged_interacts = [conv_new_interacts]
        merged = pd.concat(merged_interacts, ignore_index=True)
        self.interacts = merged

    def train_model(self, nepochs=None, nsteps=None):
        # if hasattr(self, "trainer"):
        #     print("ah shit here we go again")
        #     self.update_trainer_with_interacts(self.interacts)  # omg wtf rewrite
        # else:
        #     print("creating trainer")
        #     self.trainer = self._create_trainer(self.model)

        self.trainer.fit(nepochs, nsteps)

    def get_recommends(self, users):
        users = self.preprocessor.preprocess_users(users)
        all_item_idxs = self.item_conv.get_all_idxs()
        proc_item_idxs = self.preprocessor.preprocess_items(all_item_idxs)
        all_items_preds = self.trainer.get_recommends_for_users(users, proc_item_idxs)  # omg wtf
        user_preds = []
        for user in all_items_preds:
            user_conv = self.item_conv.get_ids(*user)
            user_preds.append(user_conv)
        return user_preds

    def _load_model_and_converters(self):
        model, (users_ids, items_ids) = self.model_saver.load(self.model_name)
        user_conv = id_idx_converter.IdIdxConverter(*users_ids)
        item_conv = id_idx_converter.IdIdxConverter(*items_ids)
        return model, user_conv, item_conv

    def _convert_interacts_add_to_convs(self, interacts):
        users = interacts[self.user_colname]
        items = interacts[self.item_colname]
        print("converting", items)
        interacts[self.user_colname] = self.user_conv.add_ids_get_idxs(*users)
        interacts[self.item_colname] = self.item_conv.add_ids_get_idxs(*items)
        return interacts

    def _update_trainer_with_interacts(self, interacts):
        users = interacts[self.user_colname]
        items = interacts[self.item_colname]
        number_of_new_users = self.user_conv.count_unknown(*users)
        number_of_new_items = self.item_conv.count_unknown(*items)

        self.trainer.add_users(number_of_new_users)
        self.trainer.add_items(number_of_new_items)

    def _create_trainer(self, model):
        dataset = self.dataloader_builder(self.interacts)
        trainer = trainers.Trainer(model, dataset, self.preprocessor, **self.train_kwargs)
        return trainer
