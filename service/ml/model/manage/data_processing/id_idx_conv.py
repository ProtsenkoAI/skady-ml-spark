from collections import OrderedDict
from bidict import bidict


class IdIdxConv:
    def __init__(self, ids=None):
        # self.id2idx = bidict({})
        # self.idx2id = self.id2idx.inverse
        self.ids = []
        # self.max_idx = 0
        if not ids is None:
            self.add_ids(*ids)

    def get_idxs(self, *ids):
        # idxs = [self.id2idx[int(id)] for id in ids]
        idxs = [self.ids.index(int(id)) for id in ids]
        return idxs

    def get_ids(self, *idxs):
        ids = [self.ids[idx] for idx in idxs]
        return ids

    def check_contains(self, obj_id):
        return obj_id in self.ids

    def get_all_ids(self):
        return sorted(self.ids)

    def get_all_idxs(self):
        return list(range(len(self.ids)))

    def delete_by_ids(self, ids):
        for obj_id in ids:
            self.ids.remove(obj_id)

    def count_unknown(self, *ids):
        count = 0
        for obj_id in set(ids):
            # if obj_id not in self.id2idx.keys():
            if obj_id not in self.ids:
                count += 1
        return count

    def add_ids(self, *ids):
        for obj_id in ids:
            # if obj_id not in self.id2idx.keys():
            if obj_id not in self.ids:
                idx = self._get_next_idx()
                # self.id2idx[obj_id] = idx
                # self.idx2id[idx] = obj_id
                self.ids.append(obj_id)

    def dump(self):
        return self.get_all_ids()

    @classmethod
    def load(cls, data):
        return cls(data)

    def _get_next_idx(self):
        # returned =self.max_idx
        # self.max_idx += 1
        # return returned
        # return len(self.id2idx) + 1
        return len(self.ids) + 1
