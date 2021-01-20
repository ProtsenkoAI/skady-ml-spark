def get_user_id(session, name_or_id):
    if isinstance(name_or_id, int):
        return name_or_id
    resp = session.method("utils.resolveScreenName", values={"screen_name": name_or_id})
    return resp["object_id"]
