from pyspark.sql import SparkSession
from abc import abstractmethod, ABC

from pyspark.sql.types import StructType
from pyspark.sql import DataFrame
from .data_managers import SparkStreamingDataManager
from ..expose.obtainer import Obtainer

Stream = DataFrame


class SparkObtainer(Obtainer, ABC):
    def __init__(self, stream_type, stream_path, app_name, schema: StructType, logging_level: str):
        """
        Takes spark stream and prepares it to fit
        """
        self.stream = self._get_stream(stream_path, stream_type, app_name, schema, logging_level)

    def _get_stream(self, pth, stream_type, app_name, schema: StructType, logging_level) -> Stream:
        print("before stream")
        spark = SparkSession \
            .builder \
            .appName(app_name).config("spark.scheduler.mode", "FAIR") \
            .getOrCreate()
        spark.sparkContext.setLogLevel(logging_level)

        if stream_type == "csv":
            stream_reader = spark.readStream
            data_stream = stream_reader.csv(pth, schema=schema)
        else:
            raise ValueError(f"The type of data_stream is not supported: {stream_type}")
        return data_stream

    def get_data(self) -> SparkStreamingDataManager:
        return SparkStreamingDataManager(self.stream)
