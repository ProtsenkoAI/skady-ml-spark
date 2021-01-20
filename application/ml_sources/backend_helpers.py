import vk_api


def create_session(access_token):
    return vk_api.VkApi(token=access_token)


def get_user_id(session, name_or_id):
    try:
        return int(name_or_id)
    except ValueError:
        resp = session.method("utils.resolveScreenName", values={"screen_name": name_or_id})
        return resp["object_id"]


# def isnumber(inputString):
#     return all(char.isdigit() for char in inputString)
