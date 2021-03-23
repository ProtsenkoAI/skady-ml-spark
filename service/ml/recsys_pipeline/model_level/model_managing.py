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

    def update_with_interacts(self, interacts):
        # TODO: remove update with interacts and replace it with simple add_user() delete_user()
        self.processor.update(interacts)
        max_user_idx, max_item_idx = self.processor.get_nusers_nitems()
        self._scale_model_if_needed(max_user_idx, max_item_idx)
        self.processor.update(interacts)

    def save(self, saver):
        name = saver.save(self.model, self.processor)
        return name

    @classmethod
    def from_save(cls, saver, name):
        model, processor = saver.load(name)
        return cls(model, processor)

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor

    # TODO: remove this ugly system of adding users and replace it with simple add_user() delete_user() methods
    def _scale_model_if_needed(self, max_user_idx, max_item_idx):
        model_kwargs = self.model.get_init_kwargs()
        nusers, nitems = model_kwargs["nusers"], model_kwargs["nitems"]
        new_users_needed = max(max_user_idx - nusers, 0)
        new_items_needed = max(max_item_idx - nitems, 0)
        if new_users_needed:
            self.model.add_users(new_users_needed)
        if new_items_needed:
            self.model.add_items(new_items_needed)

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
