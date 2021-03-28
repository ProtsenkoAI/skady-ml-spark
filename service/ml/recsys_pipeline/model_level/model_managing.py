class ModelManager:
    # TODO: eliminate save and from_save methods (because spark loader is working both with manager
    #   and trainer attributes and the abstraction goes crazy)
    def __init__(self, model, processor):
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

    # TODO: refactor savers so can use manager_saver.save() and not to expose attributes like processor
    def get_processor(self):
        return self.processor

    def get_all_users(self):
        return self.processor.get_all_items()

    def add_user(self, user_id):
        # TODO: deal with user/item is the same in Skady problem (need more generic methods etc.)
        self.processor.add_user(user_id)
        self.processor.add_item(user_id)
        self.model.add_users(1)
        self.model.add_items(1)

    def delete_user(self, user_id):
        self.model.delete_users(user_id)
        self.model.delete_items(user_id)

        user_idx_in_users_list = self.processor.convert_users(user_id)[0]
        item_idx_in_items_list = self.processor.convert_items(user_id)[0]
        self.processor.delete_users(user_idx_in_users_list)
        self.processor.delete_items(item_idx_in_items_list)
