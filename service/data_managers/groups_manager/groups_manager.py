class GroupsDataManager:
    def add(self, user_id, groups):
        """
        Adds pair of user_id -> groups to mongo object
        :param user_id: int, system user_id as it's stored at main backend
        :param groups: list with users_ids
        """
        ...

    def get_users_groups(self):
        """
        Returns groups by user_id
        """
        ...

    def remove(self, user_id):
        """
        Deletes pair (user_id, groups) from mongo object
        """
        ...
