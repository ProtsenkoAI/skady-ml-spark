# TODO: test time to load and save model
# TODO: check how the sparktorch repo is implemented on the github for insights
from train_eval.updating_weights.weights_updater import WeightsUpdater
from .trainer import Trainer
from model_level import ModelManager
from data.obtain import StreamDataManager

# TODO: refactor types


class StreamingTrainer(Trainer):
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

    def fit(self, manager: ModelManager, stream_data_manager: StreamDataManager):
        def fit_on_batch(batch, batch_idx):
            self._fit_batch_save_model(batch)

        self.spark_saver.save(manager, self.weights_updater, self.loader_builder)
        # self.fit_queue = data.writeStream.outputMode("append").foreachBatch(fit_on_batch).start()
        stream_data_manager.apply_whenever_data_recieved(fit_on_batch)
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
            weights_updater.fit_with_batch(manager, minibatch)
        self.spark_saver.save(manager, weights_updater, loader_builder)
