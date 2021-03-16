import os
import uuid
import time

from pyspark import SparkConf, SparkContext
from pyspark.sql import functions, DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType

from util import read_config
conf = read_config()


spark = SparkSession \
    .builder \
    .appName("app") \
    .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false") \
    .getOrCreate()


stream_reader = spark.readStream
cols = ["user_actor_id", "user_proposed_id", "label"]
row_schema = StructType([
    StructField(cols[0], IntegerType()),
    StructField(cols[1], IntegerType()),
    StructField(cols[2], BooleanType()),
])
data_stream = stream_reader.format("csv").load(
    os.path.join(conf["data_dir"], conf["incoming_fit_stream"]["name"]), schema=row_schema
    )

UNUSED_PATH = os.path.join(conf["data_dir"], conf["saving_unused_stream"]["name"])
os.makedirs(UNUSED_PATH, exist_ok=True)
unused_stream = stream_reader.format("csv").load(
    os.path.join(UNUSED_PATH, "*/*.csv"), schema=row_schema
    )

def set_watermark(df, timecol="timestamp") -> DataFrame:
    return df.withColumn(timecol, functions.current_timestamp()).withWatermark(timecol, "2 seconds")


def rows_to_features_and_labels(df):
    features = ["user_actor_id", "user_proposed_id"]
    features_as_arr = functions.array([df[col] for col in features])
    df = df.withColumn("features", features_as_arr)
    df = df.drop(*features)
    return df


data_stream = set_watermark(data_stream)
unused_stream = set_watermark(unused_stream)

unioned = set_watermark(data_stream.union(unused_stream))
#user_inters_count = unioned.groupBy(["user_actor_id", "timestamp"]).count()
user_inters_count = unioned.groupBy(functions.window("timestamp", "2 seconds"), "user_actor_id").count()
fit_users_df = set_watermark(user_inters_count[user_inters_count["count"] >= 10])
fit_inters = set_watermark(unioned.join(fit_users_df, on=["user_actor_id", "timestamp"]).drop("count"))

not_fit_users_df = set_watermark(user_inters_count[user_inters_count["count"] < 10])
not_fit_inters = set_watermark(unioned.join(fit_users_df, on=["user_actor_id", "timestamp"]).drop("count"))



# fit_inters_timestamp = fit_inters["timestamp"]
# fit_inters_right_col_order = fit_inters.drop("timestamp").withColumn("timestamp", fit_inters_timestamp)

print("unioned", unioned)
print("fit_inters", fit_inters)
# unused_inters = unioned.select("user_actor_id", "user_proposed_id", "label").subtract(fit_inters.select("user_actor_id", "user_proposed_id", "label"))
unused_inters = not_fit_inters

fit_data = set_watermark(fit_inters.transform(rows_to_features_and_labels))


def print_df(df, batch_id, name="some df"):
    print(name, batch_id, df.take(20))


def save_csv_random_name(df, batch_id):
    print("writing", df)
    path2csv = os.path.join(UNUSED_PATH, f"{uuid.uuid4()}.csv")
    df.write.format('csv').option('header', False).mode('overwrite').save(path2csv)


streaming_query = fit_data.writeStream.outputMode(
    "Append").foreachBatch(lambda *args: print_df(*args, name="fit")).start()

unused_query = unused_inters.writeStream.outputMode(
    "append").foreachBatch(save_csv_random_name).foreachBatch(lambda *args: print_df(*args, name="unused")).start()
#
# fit_users_df.writeStream.outputMode("complete").foreachBatch(lambda *args: print_df(*args, name="fit_users")).start()
#
# user_inters_count.writeStream.outputMode("complete").foreachBatch(lambda *args: print_df(*args, name="user_inters_count")).start()
#
# fit_inters.writeStream.outputMode("append").foreachBatch(lambda *args: print_df(*args, name="fit_inters_before")).start()
#
#
# unioned.writeStream.outputMode("append").foreachBatch(lambda *args: print_df(*args, name="unioned")).start()

unused_query.awaitTermination()
streaming_query.awaitTermination()
