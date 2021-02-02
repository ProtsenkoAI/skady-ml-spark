from data_transform import id_idx_conv, preprocessing


class ModelAssistant:
    def __init__(self):
        self.user_conv = id_idx_conv.IdIdxConverter()
        self.item_conv = id_idx_conv.IdIdxConverter()
        self.preproc = preprocessing.TensorCreator(device="cpu")

        self.user_colname = "user_id"
        self.item_colname = "anime_id"

    def preproc_then_forward(self, features):
        users, items = features
        raise NotImplementedError

    def get_recommends(self, ):
        raise NotImplementedError

    def update_with_new_interacts(self, interacts):
        raise NotImplementedError

    def preprocess(self, features, labels):
        raise NotImplementedError

    def postprocess(self, preds):
        raise NotImplementedError
