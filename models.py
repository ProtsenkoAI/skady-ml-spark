from pydantic import BaseModel, Field
from pymongo import MongoClient
from bson import ObjectId

client = MongoClient()
db = client["db"]


class PyObjectId(ObjectId):

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid objectid')
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='string')


class MatrixElement(BaseModel):
    firstUser: str
    secondUser: str
    valueMatch: float
    trackId: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }


class RequestBodyUserMatrix(BaseModel):
    userId: str
    trackId: str


class RequestBodyUserVkGroups(BaseModel):
    userId: str


