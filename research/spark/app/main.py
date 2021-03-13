from fastapi import FastAPI
from sources.ml.spark import start_auto_fitting


def main():
    start_auto_fitting()

    # bruh do it in your proper way
    api = FastAPI()
    set_api_endpoints(api)
    globals()["api"] = api


def set_api_endpoints(api: FastAPI):
    """Sets all endpoints (previously these were set in main.py)"""
    # TODO: add endpoint get_recommends(user_id: int) -> List[RecommendedUserId]
    # raise NotImplementedError
    ...


if __name__ == "__main__":
    main()
