import os
import uuid
import pandas as pd
os.environ["SPARK_HOME"] = "/home/gldsn/my_apps/spark-3.1.1-bin-hadoop2.7"

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, FloatType
from pyspark.streaming import DStream, StreamingContext

spark = SparkSession \
    .builder \
    .appName("app") \
    .getOrCreate()

stream_reader = spark.readStream
cols = ["user_actor_id", "user_proposed_id", "label"]
row_schema = StructType([
    StructField(cols[0], IntegerType(), nullable=False),
    StructField(cols[1], IntegerType(), nullable=False),
    StructField(cols[2], BooleanType(), nullable=False),
])
data_stream = stream_reader.format("csv").load(
    "/home/gldsn/Projects/skady-ml/research/spark/mock_data/", schema=row_schema
    )

UNUSED_PATH = "/home/gldsn/Projects/skady-ml/research/spark/unused_mock_data/"
os.makedirs(UNUSED_PATH, exist_ok=True)
unused_stream = stream_reader.format("csv").load(
    os.path.join(UNUSED_PATH, "*/*.csv"), schema=row_schema
    ).select(cols)
# TODO: join data_stream with unused_stream


def printwithcomment(batch, idx):
    print("unused!", batch)


inp_unused_query = unused_stream.writeStream.format("console").foreachBatch(printwithcomment).start()
# inp_unused_query.awaitTermination()


def join_with_same_user_id(key, group):
    unused = _load_or_create_unused_inters()
    unused_user_inters = unused.filter(unused["user_actor_id"] == key[0])
    # group.insert(0, "user_actor_id", key[0])
    union = group.append(unused_user_inters)
    union.columns = ["user_actor_id", "user_proposed_id", "label"]
    return union


def _load_or_create_unused_inters():
    if "unused_interacts" not in globals():
        new_df = pd.DataFrame(columns=cols)
        globals()["unused_interacts"] = new_df
    return globals()["unused_interacts"]


def rows_to_features_and_labels(df):
    return df


grouped = data_stream.groupBy("user_actor_id")
unioned = grouped.applyInPandas(join_with_same_user_id, schema=row_schema)
user_inters_count = unioned.groupBy("user_actor_id").count()
fit_users = user_inters_count.filter(user_inters_count["count"] >= 10)["user_actor_id"]

fit_row_mask = unioned["user_actor_id"].isin(fit_users)
fit_inters = unioned.filter(fit_row_mask)
unused_inters = unioned.filter(fit_row_mask == 0)

fit_data = fit_inters.transform(rows_to_features_and_labels)


def print_df(df, batch_id):
    print("Batch processed:", df, batch_id)
    print(df.take(5))


def save_csv_random_name(df, batch_id):
    path2csv = UNUSED_PATH + f"{uuid.uuid4()}.csv"
    df.write.format('csv').option('header', False).mode('overwrite').save(path2csv)


unused_query = unused_inters.writeStream.outputMode("append").foreachBatch(save_csv_random_name).start()
# path2csv = UNUSED_PATH + f"{uuid.uuid4()}.csv"
# unused_query = unused_inters.writeStream.format("csv").mode("overwrite").save(path2csv)
streaming_query = fit_data.writeStream.outputMode("append").foreachBatch(print_df).start()

unused_query.awaitTermination()
streaming_query.awaitTermination()
#
# query = data_stream \
#     .writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .start()
#
# query.awaitTermination()
