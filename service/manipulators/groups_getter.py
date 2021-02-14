import vk_api
from ..ml_sources.data_processing.vk_parsing import helpers
from ..ml_sources.data_processing.vk_parsing.obtainers import vk_obtainers
from ..ml_sources.data_processing.vk_parsing.retrievers import vk_retrievers
from ..ml_sources.data_processing.vk_parsing.exceptions import VkRequestException


class GroupsGetter:
    def __init__(self):
        self.default_creds = ["+79898797278", "63pJnPGj1"]
        self.groups_obtainer = vk_obtainers.VkGroupsObtainer()

    def get_groups(self, vk_link, access_token=None):
        self.session = self._create_session(access_token)
        vk_id = helpers.get_user_id(self.session, vk_link)
        user_retriever = vk_retrievers.ObjectRetriever(self.session, vk_id)
        try:
            return user_retriever.get(self.groups_obtainer)
        except VkRequestException:
            return None

    def _create_session(self, token):
        if token is None:
            session = vk_api.VkApi(*self.default_creds)
        else:
            session = vk_api.VKApi(token=token)
        session.auth()
        return session
