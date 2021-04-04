from warnings import warn
from .recsys_torch_model import RecsysTorchModel


class ModelManager:
    # TODO: refactor savers so can use manager_saver.save() and not to expose attributes like processor
    def __init__(self, model: RecsysTorchModel, processor):
        self.model = model
        self.processor = processor

    def preproc_forward(self, features, labels=None):
        proc_features = self.processor.preprocess_features(features)
        preds = self.model.forward(*proc_features)
        if not labels is None:
            proc_labels = self.processor.preprocess_labels(labels)
            return preds, proc_labels
        return preds

    def preproc_forward_postproc(self, features):
        preds = self.preproc_forward(features)
        return self.processor.postproc_preds(preds)

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor

    def get_all_users(self):
        return self.processor.get_all_items()

    def delete_user(self, user_id):
        if self.processor.check_user_in_model(user_id):
            user_idx_in_users_list = self.processor.convert_users(user_id)[0]
            item_idx_in_items_list = self.processor.convert_items(user_id)[0]

            self.processor.delete_users(user_id)
            self.processor.delete_items(user_id)

            self.model.delete_users(user_idx_in_users_list)
            self.model.delete_items(item_idx_in_items_list)
        else:
            warn(f"Tried to delete user {user_id} from model, but the model doesn't contain it")

    def add_user(self, user_id):
        if not self.processor.check_user_in_model(user_id):
            self._add_user_to_model()
            self._add_user_to_processor(user_id)
        else:
            warn(f"Tried to add user {user_id} to model, but the model already contains it")

    def _add_user_to_model(self):
        self.model.add_users(nb_users=1)
        self.model.add_items(nb_items=1)

    def _add_user_to_processor(self, user_id):
        self.processor.add_user(user_id)
        self.processor.add_item(user_id)
