from typing import List
from ..expose import RecommendsPostprocessor
from model.expose.types import User


class RecommendsPostprocessorImpl(RecommendsPostprocessor):
    # TODO: implement non-implemented methods
    def __init__(self):
        ...

    def process(self, user: User, raw_recommends: List[User]):
        duplicates = self.find_already_recommended(raw_recommends)
        duplicates_and_user_himself = duplicates + [user]
        new_users_to_recommend = [user for user in raw_recommends if user not in duplicates_and_user_himself]
        users_with_appropriate_study_time = self.keep_only_candidates_matching_by_study_time(user,
                                                                                             new_users_to_recommend)
        # TODO: maybe add other rules
        return users_with_appropriate_study_time

    def find_already_recommended(self, raw_recommends: List[User]) -> List[User]:
        # TODO
        return []

    def keep_only_candidates_matching_by_study_time(self, user: User, recommends: List[User]) -> List[User]:
        # TODO
        return recommends
