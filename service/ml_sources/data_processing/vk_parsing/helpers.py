import re


def get_user_id(session, name_or_id):
    if isinstance(name_or_id, int):
        return name_or_id
    resp = session.method("utils.resolveScreenName", values={"screen_name": name_or_id})
    return resp["object_id"]


def get_vk_id_from_link(link: str):
    if re.search("vk.com/id", link) is None:
        return link.split("vk.com/")[-1]
    else:
        return int(link.split("vk.com/id")[-1])