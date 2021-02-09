class ModelAssistant:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def preproc_then_forward(self, features):
        proc_features = self.processor.preprocess_features(features)
        return self.model.forward(*proc_features)

    def preproc_labels(self, labels):
        return self.processor.preprocess_labels(labels)

    def get_model(self):
        return self.model

    def get_all_items(self):
        conv = self.processor.get_item_conv()
        return conv.get_all_ids()

    def get_model_init_kwargs(self):
        return self.model.get_init_kwargs()

    def get_convs(self):
        user_conv = self.processor.get_user_conv()
        item_conv = self.processor.get_item_conv()
        return user_conv, item_conv

    def update_with_interacts(self, interacts):
        new_users_count, new_items_count = self.processor.count_unknown_users_items(interacts)
        self._scale_model_if_needed(new_users_count, new_items_count)
        self.processor.update(interacts)

    def _scale_model_if_needed(self, max_user_idx, max_item_idx):
        model_kwargs = self.get_model_init_kwargs()
        nusers, nitems = model_kwargs["nusers"], model_kwargs["nitems"]
        new_users_needed = max(max_user_idx - nusers, 0)
        new_items_needed = max(max_item_idx - nitems, 0)
        if new_users_needed:
            self.model.add_users(new_users_needed)
        if new_items_needed:
            self.model.add_items(new_items_needed)
