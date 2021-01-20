from .base_user_manipulator import BaseUserManipulator
import numpy as np


class MatchesDataGetter(BaseUserManipulator):
    def __init__(self, users_ids_manager, matrix_manager):
        groups_manager = None
        super().__init__(users_ids_manager, matrix_manager, groups_manager)

    def get_top_n_matches(self, user_id, n, access_token):
        self.create_session(access_token)
        # user_idx = self.users_ids_manager.get_indexes([user_id])
        row = self.matrix_manager.get_row(user_id)
        sorted_matches = self.sort_row(row)
        top_n = sorted_matches[:n]
        # top_ids = self.users_ids_manager.get_ids(*top_n_idxs)
        return top_n

    def sort_row(self, row):
        return self._argsort_with_randomizer_of_equals(row)

    def _argsort_with_randomizer_of_equals(self, vals):
        """Argsorting with descending, but if multiple values are equal, then this func will randomize
        position of indexes"""
        randomizer_for_equals = np.random.random(len(vals))

        sorted_idxs = np.lexsort((randomizer_for_equals, vals))
        return sorted_idxs.tolist()[::-1]
