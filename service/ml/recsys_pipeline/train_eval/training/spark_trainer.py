# TODO: test time to load and save model
# TODO: maybe learn from tests structure of sparttorch project on github
# TODO: finish type hints

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StringType, StructField
from collections import namedtuple
from .weights_updater import WeightsUpdater

# TODO: delete reading config here and move it to top level

# TODO: refactor types
SparkRes = DataFrame  # TODO: move to types

SerializablePackagedModel = namedtuple("SerializablePackagedModel",
                                       ["model", "optimizer_class", "criterion", "optim_params"])

FitBatchRes = DataFrame


class SparkTrainer:
    # TODO: check that serialises well
    # TODO: check spark_saver serialises well
    def __init__(self, loader_builder, spark_saver, lr=1e-4):
        self.loader_builder = loader_builder
        self.spark_saver = spark_saver
        self.weights_updater = WeightsUpdater(lr)
        self.fit_queue = None

        self.fit_out_schema = StructType([
            StructField("batch_fit_out", StringType())
        ])

    def fit(self, manager, data: DataFrame):
        def fit_on_batch(batch, batch_idx):
            self._fit_batch_save_model(batch)

        self.spark_saver.save(manager, self.weights_updater)

        self.fit_queue = data.writeStream.outputMode("append").foreachBatch(fit_on_batch).start()
        self.fit_queue.awaitTermination()

    def stop_fit(self):
        # TODO: test
        if self.fit_queue is None:
            raise RuntimeError("fit function was not called yet so can't stop fit")
        self.fit_queue.stop()

    def _fit_batch_save_model(self, batch):
        manager, weights_updater = self.spark_saver.load()
        weights_updater.fit_with_batch(manager, batch)
        self.spark_saver.save(manager, weights_updater)
