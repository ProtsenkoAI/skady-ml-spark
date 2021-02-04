import collections


class IdIdxConverter:
    """Service class for model manager. It's needed because model needs indexes
    of users and items, but server feeds user ids those don't go in order"""
    def __init__(self, *ids):
        self.idx2id = {}
        self.id2idx = {}
        self.max_idx = 0
        self.add_ids(*ids)

    def get_all_ids(self):
        ordered_idx2id = collections.OrderedDict(self.idx2id)
        ids_sorted_by_idx = list(ordered_idx2id.values())
        return ids_sorted_by_idx

    def get_all_idxs(self):
        return list(self.idx2id.keys())

    def count_unknown(self, *ids):
        count = 0
        for obj_id in ids:
            if obj_id not in self.id2idx.keys():
                count += 1
        return count

    def add_ids_get_idxs(self, *ids):
        self.add_ids(*ids)
        return self.get_idxs(*ids)

    def add_ids(self, *ids):
        for obj_id in ids:
            if obj_id not in self.id2idx:
                idx = self._get_next_idx()
                self.id2idx[obj_id] = idx
                self.idx2id[idx] = obj_id

    def get_idxs(self, *ids):
        idxs = [self.id2idx[int(id)] for id in ids]
        if len(idxs) == 1:
            return idxs[0]
        return idxs

    def get_ids(self, *idxs):
        ids = [self.idx2id[int(idx)] for idx in idxs]
        if len(ids) == 1:
            return ids[0]
        return ids

    def _get_next_idx(self):
        returned = self.max_idx
        self.max_idx += 1
        return returned
