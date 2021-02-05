class DataProcessor:
    def __init__(self, user_conv, item_conv):
        self.user_conv = user_conv
        self.item_conv = item_conv

    def get_user_conv(self):
        return self.user_conv

    def get_item_conv(self):
        return self.item_conv