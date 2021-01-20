import os
from flask import Flask
from .endpoints import endpoints
from . import ml_sources


def create_app(test_config=None):
    # Question: how flask receives this function from module?
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )
    app.register_blueprint(endpoints.get_bp())
    return app