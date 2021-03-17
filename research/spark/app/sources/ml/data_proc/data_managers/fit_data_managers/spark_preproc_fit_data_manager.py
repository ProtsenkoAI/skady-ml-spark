from pyspark.sql import DataFrame, functions

from .base_class import FitDataManager
from ....spark.consts import Types


class SparkStreamingFitDataManager(FitDataManager):
    def __init__(self, fit_nsamples_per_user_threshold=10):
        self.n_samples_threshold = fit_nsamples_per_user_threshold
        self.unused_interacts = None

    def prepare_for_fit(self, raw_fit_data: Types.RawData) -> Types.FitData:
        """from dataframe with many columns to dataframe with features and labels"""
        def rows_to_features_and_labels(df: DataFrame) -> Types.FitData:
            features = ["user_actor_id", "user_proposed_id"]
            features_as_arr = functions.array([df[col] for col in features])
            df = df.withColumn("features", features_as_arr)
            df = df.drop(*features)
            return df

        return raw_fit_data.transform(rows_to_features_and_labels)
