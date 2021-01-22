from fastapi import FastAPI

from models import db, MatrixElement, RequestBodyUserMatrix, RequestBodyUserVkGroups
from service.data_managers.groups_manager import GroupsDataManager

app = FastAPI()
groups_data_manager = GroupsDataManager()


@app.get('/get_user_groups')
async def groups_list(req_body: RequestBodyUserVkGroups):
    return await groups_data_manager.get_users_groups(req_body.userId)

 
@app.get('/get_match_list')
async def get_match_list(req_body: RequestBodyUserMatrix):
    matrix = await groups_data_manager.get_match_values(req_body.userId, req_body.trackId)
    return {'userMatches': matrix}


@app.post('/add_new_user')
async def create_matrix(req_body: RequestBodyUserMatrix):
    matrix = await groups_data_manager.add(req_body.userId, req_body.trackId)
    return matrix
