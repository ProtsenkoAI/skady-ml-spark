class MatchEstimator:
    def get_match_val(self, groups1, groups2):
        if self._check_none_input(groups1, groups2):
            return None
        groups1, groups2 = set(groups1), set(groups2)

        intersect_elems = groups1.intersection(groups2)
        intersection = len(intersect_elems)
        union_elems = groups1.union(groups2)
        union = len(union_elems)

        return intersection / union

    def _check_none_input(self, *inputs):
        for inp in inputs:
            if inp is None:
                return True
        return False