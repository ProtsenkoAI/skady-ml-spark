# TODO: handle invalid inputs
from . import endpoint_helpers
from flask import Blueprint, request
import json
bp = Blueprint('matching', __name__, url_prefix='/matching')


def get_bp(): return bp


@bp.route("/user_groups", methods=('POST',))
def get_user_groups():
    """
    Returns list of user groups
    :param: user_vk: int (vk id) or string (vk short_name)
    :param access_token: str with access token that gives permission to acces user's page info
    :return:
        'user_groups': list of user gorups
    """
    user_vk = request.form["user_vk"]
    # if endpoint_helpers.isnumber(user_vk):
    #     user_vk = int(user_vk)
    access_token = request.form["access_token"]

    user_groups = endpoint_helpers.get_user_groups(user_vk, access_token)
    return {"user_groups": user_groups}


@bp.route("/match_values", methods=("POST",))
def get_matches_of_user():
    """
    Returns list with match values
    :param user_data: list with user groups
    :param others_data: iterable with lists of other user groups
    :return:
        'match_values': list of len = len(others_data) containing floats (matching values)
    """
    # TODO: rewrite to get list of candidates
    user_groups = json.loads(request.form["user_data"])
    others_data = json.loads(request.form["others_data"])
    return {"match_values": endpoint_helpers.iou_with_others(user_groups, others_data)}


@bp.route("/sorted_match_vals", methods=["POST"])
def sort_match_vals():
    """
    Returns indexes of input list sorted by element values
    :param match_values: list of matches with other users, that has to be matched
    :return:
        'sorted_indexes': list of len = len(match_vals) containing ints (sorted indexes)

    Example: match_values = [1, 9, 3], then function will return [0, 2, 1]
    """
    match_values = json.loads(request.form["match_values"])
    sorted_vals = endpoint_helpers.argsort_with_randomizer_of_equals(match_values)
    return {"sorted_indexes": sorted_vals}
