from collections import OrderedDict


class IdIdxConv:
    def __init__(self, ids=None):
        self.idx2id = {}
        self.id2idx = {}
        self.max_idx = 0
        if not ids is None:
            self.add_ids(*ids)

    def get_idxs(self, *ids):
        idxs = [self.id2idx[int(id)] for id in ids]
        return idxs

    def get_ids(self, *idxs):
        ids = [self.idx2id[int(idx)] for idx in idxs]
        return ids

    def delete_by_ids(self, ids):
        for obj_id in ids:
            idx = self.id2idx[obj_id]
        raise NotImplementedError

    def get_all_ids(self):
        ordered_idx2id = OrderedDict(self.idx2id)
        ids_sorted_by_idx = list(ordered_idx2id.values())
        return ids_sorted_by_idx

    def get_all_idxs(self):
        idxs = sorted(self.idx2id.keys())
        return idxs

    def count_unknown(self, *ids):
        count = 0
        for obj_id in set(ids):
            if obj_id not in self.id2idx.keys():
                count += 1
        return count

    def add_ids(self, *ids):
        for obj_id in ids:
            if obj_id not in self.id2idx.keys():
                idx = self._get_next_idx()
                self.id2idx[obj_id] = idx
                self.idx2id[idx] = obj_id

    def dump(self):
        return self.get_all_ids()

    @classmethod
    def load(cls, data):
        return cls(data)

    def _get_next_idx(self):
        returned = self.max_idx
        self.max_idx += 1
        return returned
