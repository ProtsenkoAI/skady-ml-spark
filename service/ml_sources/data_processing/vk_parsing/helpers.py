import re
from .exceptions import VkRequestException


def get_user_id(session, link):
    root_url = "vk.com"
    root_url_with_id = root_url + "/id"
    if re.search(root_url_with_id, link):
        user_id = link.split(root_url_with_id)[-1]
        return int(user_id)
    else:
        user_shortname = link.split(root_url)[-1][1:]
        try:
            return get_user_id_from_shortname(session, user_shortname)
        except VkRequestException:
            raturn None


def get_user_id_from_shortname(session, shortname):
    resp = session.method("utils.resolveScreenName", values={"screen_name": shortname})
    user_id = resp["object_id"]
    return user_id
