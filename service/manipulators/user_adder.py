import vk_api

from app.main.service.ml_sources.data_processing.vk_parsing import helpers
from app.main.service.ml_sources.data_processing.vk_parsing.obtainers import vk_obtainers
from app.main.service.ml_sources.data_processing.vk_parsing.retrievers import vk_retrievers
from app.main.service.manipulators.base_user_manipulator import BaseUserManipulator


class UserAdder(BaseUserManipulator):
    def add_user(self, vk_link, user_id, access_token):
        self.session = self.create_session(access_token)
        vk_id = helpers.get_user_id(self.session, vk_link)
        self.users_ids_manager.append(user_id)

        groups = self.get_groups(vk_id)

        others_groups = self.groups_manager.get_users_groups()
        self.groups_manager.add(user_id, groups)

        match_vals = self.calc_match(groups, others_groups)
        self.matrix_manager.add_row_and_col(match_vals)

    def get_groups(self, vk_id):
        groups_obtainer = vk_obtainers.VkGroupsObtainer()
        user_retriever = vk_retrievers.ObjectRetriever(self.session, vk_id)
        return user_retriever.get(groups_obtainer)

    def calc_match(self, user_groups, others_groups):
        vals = []
        for groups in others_groups:
            val = self.calc_match_val(user_groups, groups)
            vals.append(val)
        return vals

    def calc_match_val(self, groups1, groups2):
        groups1, groups2 = set(groups1), set(groups2)

        intersect_elems = groups1.intersection(groups2)
        intersection = len(intersect_elems)

        union_elems = groups1.union(groups2)
        union = len(union_elems)
        return intersection / union

