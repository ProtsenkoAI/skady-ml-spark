from fastapi import FastAPI
from typing import List, Tuple

import pandas as pd
from pyspark.sql import functions, Column
from pyspark.streaming.dstream import DStream
from pyspark.streaming import StreamingContext
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql import SparkSession

from util import read_config
config = read_config()


cols = ["user_actor_id", "user_proposed_id", "result"]
row_schema = StructType([
    StructField(cols[0], IntegerType(), nullable=True),
    StructField(cols[1], IntegerType(), nullable=True),
    StructField(cols[2], BooleanType(), nullable=True),
])

Interacts = DataFrame
SomeModelType = None


def main():
    start_auto_fitting()

    # bruh do it in your proper way
    api = FastAPI()
    set_api_endpoints(api)
    globals()["api"] = api


def set_api_endpoints(api: FastAPI):
    """Sets all endpoints (previously these were set in main.py)"""
    # TODO: add endpoint get_recommends(user_id: int) -> List[RecommendedUserId]
    raise NotImplementedError


def start_auto_fitting():
    # master is initialized automatically (in spark-submit mode it's taken from --master argument
    spark = SparkSession\
        .builder\
        .appName(config["app_name"])\
        .getOrCreate()

    stream_type = config["data_stream"]["type"]
    if stream_type == "text_files_dir":

        # stream_cont = StreamingContext(spark.sparkContext, batchDuration=2)
        # data_stream = stream_cont.textFileStream(config["data_stream"]["path"])
        stream_reader = spark.readStream
        data_stream = stream_reader.csv(config["data_stream"]["path"], schema=row_schema)
    else:
        raise ValueError(f"The type of data_stream is not supported: {stream_type}")

    interacts = transform_to_interacts(data_stream)
    update_users_history(interacts)
    # do_some_shit(interacts)
    fit_interacts = filter_for_fit(interacts)
    # fit_interacts = interacts.transform(filter_for_fit)
    model = get_or_load_model()
    fit_model(fit_interacts, model)
    save_model(model)


def transform_to_interacts(stream: DStream, col_names=cols) -> Interacts:
    interacts = stream
    print("interacts", interacts)
    return interacts


def update_users_history(interacts: Interacts):
    """Saves interacts to long-term history to load use it later"""
    # TODO
    pass


def filter_for_fit(new_interacts: Interacts, num_inters_thresh=10) -> Interacts:
    """
    Idea: we accumulate interacts, and choose users whose number of interacts >= threshold,
    then we delete interacts of these users from accumulated interacts.
    """
    # TODO: group up by user_actor
    joined_inters = _concat_interacts(new_interacts, func_to_load_inters=get_or_create_unused_interacts)
    chosen_users = _chose_fit_users(joined_inters, num_inters_thresh)
    # TODO: it'll not work if spark memorized copy of unused_interacts to join. Debug: that with new batches
    # unused interacts are affecting join result!!
    fit_interacts, new_unused_interacts = _split_by_users(joined_inters, chosen_users)
    save_unused_interacts(new_unused_interacts)
    return fit_interacts


def _concat_interacts(src_interacts, func_to_load_inters) -> Interacts:
    """Concats src_interacts with unused_interacts.
    The difficulty is that we can't hardcode src.union(unused), because when next batches will arrive,
    unused will not be updated because it's immutable object, so we have to load them dynamically from save inside
    function added to spark's DAG.
    """
    # TODO: change the implementation from groupby to repartition of unused and union on every partition of
    #  src_interacts
    def join_with_same_user_id(group: pd.DataFrame):
        unused = func_to_load_inters()
        group_user_id = group["user_actor_id"].iloc[0]
        unused_user_inters = unused.filter(unused["user_actor_id"] == group_user_id)
        return group.append(unused_user_inters.toPandas())

    # print("inp inters", inters)
    # print(dir(inters.groupBy("user_actor")))

    grouped = src_interacts.groupby("user_actor_id")
    return grouped.applyInPandas(join_with_same_user_id, schema=row_schema)

    # accumulated = interacts[0]
    # for next_interacts in interacts[1:]:
    #     accumulated = accumulated.union(next_interacts)
    # return accumulated


def _chose_fit_users(joined: Interacts, n_samples_threshold: int):
    # TODO: check that works properly
    user_inters_count = joined.groupBy("user_actor_id").count()
    fit_users = user_inters_count.filter(user_inters_count["count"] >= n_samples_threshold)["user_actor_id"]
    print("fit_users", fit_users)
    return fit_users


def _split_by_users(joined_inters: Interacts, users: Column) -> Tuple[Interacts, Interacts]:
    """
    Returns: Tuple[Interacts of users in users Column, Interacts of users not in users Column]
    """
    fit_row_mask = joined_inters["user_actor_id"].isin(users)
    fit_inters = joined_inters.filter(fit_row_mask)
    new_unused_inters = joined_inters.filter(fit_row_mask == 0)  # debug == 0 works properly
    return fit_inters, new_unused_inters


def get_or_create_unused_interacts() -> Interacts:
    if "unused_interacts" not in globals():
        # mif it'll create new context will cause errors, so # Attention
        spark = SparkSession.builder.getOrCreate()
        new_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), row_schema)
        globals()["unused_interacts"] = new_df
    return globals()["unused_interacts"]


def save_unused_interacts(inters: Interacts):
    globals()["unused_interacts"] = inters


def get_or_load_model() -> SomeModelType:
    raise NotImplementedError


def fit_model(inters, model: SomeModelType):
    raise NotImplementedError


def save_model(model: SomeModelType):
    raise NotImplementedError


if __name__ == "__main__":
    main()
