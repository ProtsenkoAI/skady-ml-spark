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

