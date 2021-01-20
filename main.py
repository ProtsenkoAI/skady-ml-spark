from fastapi import FastAPI

from models import db, MatrixElement

app = FastAPI()


@app.get('/matrix')
async def list_matrix():
    matrix = []
    for element in db.matrix.find():
        matrix.append(MatrixElement(**element))
    return {'matrix': matrix}


@app.post('/matrix')
async def create_matrix(matrix_element: MatrixElement):
    if hasattr(matrix_element, 'id'):
        delattr(matrix_element, 'id')
    ret = db.matrix.insert_one(matrix_element.dict(by_alias=True))
    matrix_element.id = ret.inserted_id
    return {'matrix': matrix_element}
