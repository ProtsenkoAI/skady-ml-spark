class UsersIdsDataManager:
    """This class converts main-backend user id to matching user_idx.
    For example, ml_backend recieves request with user_id = 1234132, we then convert it to user_idx = 123,
    and then using the index at our side."""
    def append(self, user_id):
        """Asigns index to user_id, adds pair (user_id, user_index) to mongo object"""
        ...

    def get_indexes(self, *user_ids):
        """list(user_ids) -> list(user_indexes)"""
        ...

    def get_ids(self, *indexes):
        """list(user_indexes) -> list(user_ids)"""
        ...

    def remove(self, idx):
        """Removes pair (user_id, idx) from mongo object"""
        ...