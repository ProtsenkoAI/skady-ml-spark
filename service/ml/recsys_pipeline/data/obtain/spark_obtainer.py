from pyspark.sql import SparkSession
import os

from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType
from pyspark.sql import DataFrame
from .data_managers import StreamingDataManager, SparkStreamingDataManager

Stream = DataFrame
FitData = StreamingDataManager

cols = ["user_actor_id", "user_proposed_id", "label"]
row_schema = StructType([
    StructField(cols[0], IntegerType(), nullable=True),
    StructField(cols[1], IntegerType(), nullable=True),
    StructField(cols[2], BooleanType(), nullable=True),
])


class SparkObtainer:
    def __init__(self, config, paths):
        """
        Takes spark stream and prepares them for fit
        """
        # TODO: not to hardcode label_colname
        self.stream_type = config["fit_stream"]["type"]
        self.stream_path = os.path.join(paths["base_path"], config["fit_stream"]["relative_path"])
        self.app_name = config["app_name"]

        self.stream = self._get_stream(self.stream_path, self.stream_type, self.app_name)

    def _get_stream(self, pth, stream_type, app_name) -> Stream:
        spark = SparkSession \
            .builder \
            .appName(app_name).config("spark.scheduler.mode", "FAIR") \
            .getOrCreate()

        if stream_type == "csv":
            stream_reader = spark.readStream
            data_stream = stream_reader.csv(pth, schema=row_schema)
        else:
            raise ValueError(f"The type of data_stream is not supported: {stream_type}")
        return data_stream

    def get_fit_data(self) -> FitData:
        return SparkStreamingDataManager(self.stream)
