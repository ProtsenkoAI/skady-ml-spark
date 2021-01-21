from main import db
from bson import ObjectId

from models import MatrixElement
from service.manipulators.groups_getter import GroupsGetter
from service.manipulators.match_estimator import MatchEstimator


class GroupsDataManager:
    async def add(self, user_id, track_id):
        """
        Adds pair of user_id -> groups to mongo object
        :param user_id: int, system user_id as it's stored at main backend
        :param track_id: str, track id
        """
        match_estimator = MatchEstimator()

        user = db.users.find_one({"_id": ObjectId(user_id)})
        users = list(db.users.find())

        for index in range(len(users)):
            if users[index]["_id"] != user["_id"]:
                match_value = match_estimator.get_match_val(user["vkGroups"], users[index]["vkGroups"])
                matrix_element = MatrixElement(**{
                    "firstUser": str(user["_id"]),
                    "secondUser": str(users[index]["_id"]),
                    "valueMatch": match_value if match_value else 0.0,
                    "trackId": track_id
                })
                db.matrix.insert_one(matrix_element.dict())

    async def get_users_groups(self, user_id):
        """
        Returns groups by user_id
        """
        user = db.users.find_one({"_id": ObjectId(user_id)})
        groups_getter = GroupsGetter()
        return groups_getter.get_groups(user["vkLink"])

    def remove(self, user_id):
        """
        Deletes pair (user_id, groups) from mongo object
        """
        ...
