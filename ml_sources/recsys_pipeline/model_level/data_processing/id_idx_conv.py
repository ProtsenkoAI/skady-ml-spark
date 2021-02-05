class IdIdxConv:
    def __init__(self, ids):
        self.ids = ids

    def dump(self):
        return [1, 2, 3]

    @classmethod
    def load(cls, data):
        return cls(data)