from fastapi import FastAPI
from sources.ml.spark import start_auto_fitting


def main():
    # TODO: make function to parse nginx log and push batches of data to fit
    start_auto_fitting()
    # bruh do API things in your proper way
    api = FastAPI()
    set_api_endpoints(api)
    globals()["api"] = api  # move api to globals() because FastAPI needs it to start app


def set_api_endpoints(api: FastAPI):
    """Sets all endpoints (previously these were set in main.py)"""
    # TODO: add endpoint get_recommends(user_id: int) -> List[RecommendedUserId]
    # raise NotImplementedError
    ...


if __name__ == "__main__":
    main()
