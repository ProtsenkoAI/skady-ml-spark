# TODO: test time to load and save model
# TODO: check how the sparktorch repo is implemented on the github for insights
# TODO: finish type hints

from pyspark.sql import DataFrame
from .weights_updater import WeightsUpdater

# TODO: delete reading config here and move it to top level

# TODO: refactor types
SparkRes = DataFrame  # TODO: move to types

FitBatchRes = DataFrame


class SparkTrainer:
    # TODO: check that serialises well
    # TODO: check for better ways for serializing strategies
    def __init__(self, loader_builder, spark_saver, lr=1e-4):
        # TODO: add epochs and steps support
        # TODO: maybe replace spark_saver with some sort of PackagedModel object that can dump and load itself
        #   (possibly using spark_saver analogue under the hood)
        self.loader_builder = loader_builder
        self.spark_saver = spark_saver
        self.weights_updater = WeightsUpdater(lr)
        self.fit_queue = None

    def fit(self, manager, data: DataFrame):
        def fit_on_batch(batch, batch_idx):
            self._fit_batch_save_model(batch)

        self.spark_saver.save(manager, self.weights_updater, self.loader_builder)
        # TODO: the code below is dependent from spark structured streaming. Make it work with any kind
        # of spark data (non-streaming too)
        self.fit_queue = data.writeStream.outputMode("append").foreachBatch(fit_on_batch).start()
        # TODO: run fitting in some thread process that can continue in background
        # self.fit_queue.awaitTermination()

    def stop_fit(self):
        # TODO: test
        if self.fit_queue is None:
            raise RuntimeError("fit function was not called yet so can't stop fit")
        self.fit_queue.stop()

    def _fit_batch_save_model(self, fit_data: DataFrame):
        # NOTE: be careful to distinct spark batches and ML minibatches concept
        print("start fitting")
        # TODO: add loader to fitter (at the moment it doesn't fit with epochs, on whole batch at once)
        manager, weights_updater, loader_builder = self.spark_saver.load()
        weights_updater.prepare_for_fit(manager)
        fit_data_df = fit_data.toPandas()
        batch_iterator = loader_builder.build(fit_data_df)
        # TODO: change iterating for batches to some custom iterator object
        # for minibatch in batch_iterator(fit_data):
        for minibatch in batch_iterator:
            print("batch", minibatch)
            weights_updater.fit_with_batch(manager, minibatch)
        print("saving")
        self.spark_saver.save(manager, weights_updater, loader_builder)
        print("end saving")
