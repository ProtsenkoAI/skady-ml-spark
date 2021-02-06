class ModelAssistant:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def get_model(self):
        return self.model

    def get_model_init_kwargs(self):
        return {"nusers": 50, "nitems": 20, "hidden_size": 5}

    def get_convs(self):
        user_conv = self.processor.get_user_conv()
        item_conv = self.processor.get_item_conv()
        return user_conv, item_conv

    def update_and_convert(self, interacts):
        new_users_count, new_items_count = self.processor.count_unknown_users_items(interacts)
        self._scale_model_if_needed(new_users_count, new_items_count)
        proc_inters = self.processor.update_and_convert(interacts)
        return proc_inters

    def _scale_model_if_needed(self, max_user_idx, max_item_idx):
        model_kwargs = self.get_model_init_kwargs()
        nusers, nitems = model_kwargs["nusers"], model_kwargs["nitems"]
        new_users_needed = max(max_user_idx - nusers, 0)
        new_items_needed = max(max_item_idx - nitems, 0)
        if new_users_needed:
            self.add_users(new_users_needed)
        if new_items_needed:
            self.add_items(new_items_needed)

    def add_users(self, nusers):
        ...

    def add_items(self, nitems):
        ...
