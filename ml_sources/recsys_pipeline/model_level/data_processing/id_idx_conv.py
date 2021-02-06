class IdIdxConv:
    def __init__(self, ids=None):
        self.idx2id = {}
        self.id2idx = {}
        self.max_idx = 0
        if not ids is None:
            self.add_ids(*ids)

    def get_idxs(self, *ids):
        idxs = [self.id2idx[int(id)] for id in ids]
        if len(idxs) == 1:
            return idxs[0]
        return idxs

    def count_unknown(self, *ids):
        count = 0
        for obj_id in ids:
            if obj_id not in self.id2idx.keys():
                count += 1
        return count

    def add_ids(self, *ids):
        for obj_id in ids:
            if obj_id not in self.id2idx:
                idx = self._get_next_idx()
                self.id2idx[obj_id] = idx
                self.idx2id[idx] = obj_id

    def dump(self):
        return [1, 2, 3]

    @classmethod
    def load(cls, data):
        return cls(data)

    def _get_next_idx(self):
        returned = self.max_idx
        self.max_idx += 1
        return returned