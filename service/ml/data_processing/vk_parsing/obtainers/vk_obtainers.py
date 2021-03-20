import vk_api
import collections
from .. import exceptions

class VkObtainer:
    """Base class for obtainers/
    Every obtainer should init get_method_name() method (obtainer uses it in request).
    """
    # NOTE: hm, maybe init is redundant and we just have to use custom params
    def __init__(self, **req_kwargs): 
        self.req_kwargs = req_kwargs

    def request(self, session, user_id):
        self.add_req_value("user_id", user_id)
        values = self.req_kwargs.copy()
        return self._request_with_values(session, values)

    def _request_with_values(self, session, values):
        try:
            return session.method(self.get_method_name(), values=values)
        except vk_api.ApiError as e: #user or group is private or smth else
            if e.code == 29:
                raise exceptions.VkRequestRateException()
            raise exceptions.VkRequestException(e, values)

    def parse(self, resp):
        raise NotImplementedError

    def get_method_name(self):
        raise NotImplementedError

    def add_req_value(self, name, value):
        self.req_kwargs[name] = value


class VkFriendsObtainer(VkObtainer):
    def request(self, session, user_id):
        return super().request(session, user_id)

    def parse(self, resp):
        return resp["items"]

    def get_method_name(self):
        return "friends.get"


class VkGroupsObtainer(VkObtainer):
    def request(self, session, user_id):
        resp = super().request(session, user_id)
        return resp

    def parse(self, resp):
        return resp["items"]

    def get_method_name(self):
        return "groups.get"


class VkPostsObtainer(VkObtainer):
    def __init__(self, *args, is_group=False, **kwargs):
        self.is_group = is_group
        super().__init__(*args, **kwargs)

    def request(self, session, owner_id):
        if self.is_group:
            owner_id = -1 * owner_id # vk_api feature

        values = self.req_kwargs.copy()
        values["owner_id"] = owner_id
        self._request_with_values(values)

    def parse(self, resp):
        items = resp["items"]
        texts = [post["text"] for post in items if post["text"] != ""]
        if len(texts) == 0:
            raise exceptions.VkParseException("no texts on wall", resp)
        return texts

    def get_method_name(self):
         return "wall.get"


class VkInfoObtainer(VkObtainer):
    def __init__(self, fields_scheme="all", *args, **kwargs):
        if fields_scheme == "all": 
            self.returned_fields_list = ["sex", "bdate", "city", "country", "contacts", 
                                "education", "universities", "schools", "status", 
                                "last_seen", "followers_count", "occupation", 
                                "relatives", "relation", "relatives", "personal", 
                                "connections", "exports", "activities", 
                                "interests", "music", "movies", "tv", "books", 
                                "games", "about", "quotes"]

        if fields_scheme == "match":
            self.returned_fields_list = ["bdate", "city", "country", "universities", 
                                        "schools",  "occupation", "relation", "personal", 
                                        "connections", "exports", "activities", 
                                        ]
        else:
            raise ValueError(f"Unimplemented filds_scheme value: {fields_scheme}")
        super().__init__(*args, **kwargs)

    def request(self, session, user_id):
        fields = ",".join(self.returned_fields_list)
        self.add_req_value("fields", fields)
        return super().request(session, user_id)

    def parse(self, resp):
        raise NotImplementedError

    def get_method_name(self):
        return "users.get"


class VkGroupsCategoriesObtainer(VkObtainer):
    def __init__(self, *args, n_most_common=None, **kwargs):
        self.n_most_common = n_most_common
        self.popped_categories = ["Открытая группа"]
        super.__init__(*args, **kwargs)

    def request(self, session, user_id):
        self.add_req_value("extended", 1)
        self.add_req_value("fields", "activity") # "activity" contains group category
        return super().request(session, user_id)

    def parse(self, resp):
        groups = resp["items"]
        activities = self._extract_groups_activities(groups)

        activities_count = collections.Counter(activities)
        # if self.n_most_common is None, uses all categories
        common_activities_count = activities_count.most_common(self.n_most_common)
        activities_count = dict(common_activities_count)
        activities_count_popped = self._pop_categories(activities_count)
        return activities_count_popped

    def _extract_groups_activities(self, groups):
        activities = []
        for group in groups:
            if "activity" in group:
                activities.append(group["activity"])
            else:
                print("group has no activity, probably it's private")

        return activities

    def _pop_categories(self, act_count):
        for cat in self.popped_categories:
            activities_count.pop(cat, None)

        return act_count

    def get_method_name(self):
        return "groups.get"


class IsPrivateObtainer(VkObtainer):
    def __init__(self, **req_kwargs):
        self.req_kwargs = req_kwargs

    def request(self, session, ids):
        ids_str = ",".join([str(id) for id in ids])
        self.add_req_value("user_ids", ids_str)
        return self._request_with_values(session, self.req_kwargs) # TODO: refactor

    def parse(self, resp):
        out = []
        for user_info in resp:
            if "deactivated" in user_info.keys():
                out.append(True)
            else:
                out.append(user_info["is_closed"])

        return out

    def get_method_name(self):
        return "users.get"
