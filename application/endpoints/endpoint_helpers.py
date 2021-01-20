from ..ml_sources.data_processing.vk_parsing.obtainers import vk_obtainers
from ..ml_sources.data_processing.vk_parsing.retrievers import vk_retrievers
from ..ml_sources import backend_helpers

import numpy as np


def iou_with_others(user_groups, others_groups):
    vals = []
    for other_user_groups in others_groups:
        iou = lists_iou(user_groups, other_user_groups)
        vals.append(iou)

    return vals


def get_user_groups(user_vk, access_token):

    session = backend_helpers.create_session(access_token)
    groups_obtainer = vk_obtainers.VkGroupsObtainer()
    user_groups = _get_user_data(user_vk, session, groups_obtainer)
    return user_groups

def _get_user_data(user_vk, session, groups_obtainer):
    # TODO: what to return if account is private?
    user_id = backend_helpers.get_user_id(session, user_vk)
    user_data = vk_retrievers.ObjectRetriever(session, user_id)
    user_groups = user_data.get(groups_obtainer)
    return user_groups


def lists_iou(lst1, lst2):
    lst1, lst2 = set(lst1), set(lst2)

    intersect_elems = lst1.intersection(lst2)
    intersection = len(intersect_elems)

    union_elems = lst1.union(lst2)
    union = len(union_elems)
    return intersection / union


def argsort_with_randomizer_of_equals(vals):
    """Argsorting with descending, but if multiple values are equal, then this func will randomize
    position of indexes"""
    randomizer_for_equals = np.random.random(len(vals))

    sorted_idxs = np.lexsort((randomizer_for_equals, vals))
    return sorted_idxs.tolist()[::-1]
