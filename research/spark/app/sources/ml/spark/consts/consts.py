from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType
from pyspark.sql import DataFrame
from pyspark import RDD
from pyspark.streaming import DStream

cols = ["user_actor_id", "user_proposed_id", "label"]
row_schema = StructType([
    StructField(cols[0], IntegerType(), nullable=True),
    StructField(cols[1], IntegerType(), nullable=True),
    StructField(cols[2], BooleanType(), nullable=True),
])


class Types:
    RawData = RDD
    FitData = DataFrame
    StreamData = DStream
