import os
import pandas as pd
os.environ["SPARK_HOME"] = "/home/gldsn/my_apps/spark-3.1.1-bin-hadoop2.7"

from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, FloatType
from pyspark.streaming import DStream, StreamingContext

spark = SparkSession \
    .builder \
    .appName("app") \
    .getOrCreate() \

stream_reader = spark.readStream
cols = ["user_actor_id", "user_proposed_id", "label"]
row_schema = StructType([
    StructField(cols[0], IntegerType(), nullable=False),
    StructField(cols[1], IntegerType(), nullable=False),
    StructField(cols[2], BooleanType(), nullable=False),
])
dataset = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))


def join_with_same_user_id(key, group):
    unused = _load_or_create_unused_inters()
    unused_user_inters = unused.filter(unused["user_actor_id"] == key[0])
    group.insert(0, "user_actor_id", key[0])
    union = group.append(unused_user_inters)
    union.columns = ["user_actor_id", "user_proposed_id", "label"]
    return union


def _load_or_create_unused_inters():
    if "unused_interacts" not in globals():
        # # mif it'll create new context will cause errors, so # Attention
        # spark = SparkSession.builder.getOrCreate()
        # new_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), row_schema)
        new_df = pd.DataFrame(columns=cols)
        globals()["unused_interacts"] = new_df
    return globals()["unused_interacts"]


grouped = dataset.groupBy("id")
unioned = grouped.applyInPandas(join_with_same_user_id, schema=row_schema)

print(unioned.collect())
