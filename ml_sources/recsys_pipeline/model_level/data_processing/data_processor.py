class DataProcessor:
    def __init__(self, user_conv, item_conv):
        self.user_conv = user_conv
        self.item_conv = item_conv

        self.user_colname = "user_id"
        self.item_colname = "anime_id"

    def get_user_conv(self):
        return self.user_conv

    def get_item_conv(self):
        return self.item_conv

    def count_unknown_users_items(self, interacts):
        users, items = self._get_users_items(interacts)
        nb_unknown_users = self.user_conv.count_unknown(*users)
        nb_unknown_items = self.item_conv.count_unknown(*items)
        return nb_unknown_users, nb_unknown_items

    def update_and_convert(self, interacts):
        users, items = self._get_users_items(interacts)
        self.user_conv.add_ids(*users)
        self.item_conv.add_ids(*items)
        interacts[self.user_colname] = self.user_conv.get_idxs(*users)
        interacts[self.item_colname] = self.item_conv.get_idxs(*items)
        return interacts

    def _get_users_items(self, interacts):
        users = interacts[self.user_colname]
        items = interacts[self.item_colname]
        return users, items