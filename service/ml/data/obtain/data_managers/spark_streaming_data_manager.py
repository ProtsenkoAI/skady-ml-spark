from typing import Optional, Callable
from pyspark.sql import DataFrame
from ...expose.streaming_data_manager import StreamingDataManager
import time


class SparkStreamingDataManager(StreamingDataManager):
    def __init__(self, stream: DataFrame):
        self.stream = stream.writeStream.outputMode("append")
        self.queries = []

    def stop_queries(self):
        for query in self.queries:
            query.stop()

    def await_all_queries(self, timeout: Optional[int] = None):
        for query in self.queries:
            if query.isActive and timeout > 0:
                start_time = time.time()

                query.awaitTermination(timeout)

                time_passed = time.time() - start_time
                timeout -= time_passed

    def apply_to_each_batch(self, obj_to_apply: Callable, kwargs: Optional[dict] = None):
        if kwargs is None:
            kwargs = {}

        def apply_to_data_batch(batch_raw: DataFrame, batch_idx: int):
            batch = batch_raw.toPandas()
            obj_to_apply(batch, **kwargs)

        query = self.stream.foreachBatch(apply_to_data_batch).start()
        self.queries.append(query)

    def apply_to_each_row(self, obj_to_apply: Callable, kwargs: Optional[dict] = None):
        if kwargs is None:
            kwargs = {}

        def apply_to_row(row_raw: DataFrame, row_idx):
            row = row_raw.toPandas()
            obj_to_apply(row, **kwargs)

        query = self.stream.foreachBatch(apply_to_row).start()
        self.queries.append(query)
