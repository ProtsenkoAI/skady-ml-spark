class MatchEstimator:
    def get_match_val(self, groups1, groups2):
        groups1, groups2 = set(groups1), set(groups2)

        intersect_elems = groups1.intersection(groups2)
        intersection = len(intersect_elems)
        union_elems = groups1.union(groups2)
        union = len(union_elems)

        return intersection / union
