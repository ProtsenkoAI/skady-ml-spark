from typing import Tuple, Union
import pandas as pd
from pyspark.sql import SparkSession, DataFrame, functions

from .base_class import FitDataManager
from ....spark.consts import Types, row_schema

# TODO: save and load unused as pandas dataframe
# TODO: test that unused updates with new batches
class SparkStreamingFitDataManager(FitDataManager):
    unused_interacts: Union[None, Types.RawData]

    def __init__(self, fit_nsamples_per_user_threshold=10):
        self.n_samples_threshold = fit_nsamples_per_user_threshold
        self.unused_interacts = None

    def union_with_saved_unused(self, new_data: Types.RawData) -> Types.RawData:
        """Unions new_data with dynamically-loaded unused_data. It allows to get updated unused_data even with static
            DAG that spark uses."""
        # TODO: change the implementation from groupby to repartition of unused and union on every partition of
        #  src_interacts
        def join_with_same_user_id(group: pd.DataFrame):
            # unused = self._load_or_create_unused_inters()
            # group_user_id = group["user_actor_id"].iloc[0]
            # unused_user_inters = unused.filter(unused["user_actor_id"] == group_user_id)
            # return group.append(unused_user_inters.toPandas())
            # TODO: uncomment function above
            return group

        grouped = new_data.groupBy("user_actor_id")
        return grouped.applyInPandas(join_with_same_user_id, schema=row_schema)

    def _load_or_create_unused_inters(self):
        if self.unused_interacts is None:
            # mif it'll create new context will cause errors, so # Attention
            spark = SparkSession.builder.getOrCreate()
            new_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), row_schema)
            self.unused_interacts = new_df
            # with open("/tmp/abc.txt", "w") as f:
            #     f.write("creating new unused interacts 'cause old weren't saved")
        print("unused", self.unused_interacts)
        return self.unused_interacts

    def prepare_for_fit(self, raw_fit_data: Types.RawData) -> Types.FitData:
        """from dataframe with many columns to dataframe with features and labels"""
        def rows_to_features_and_labels(df: DataFrame) -> Types.FitData:
            # features = ["user_actor_id", "user_proposed_id"]
            # features_as_arr = functions.array([df[col] for col in features])
            # df = df.withColumn("features", features_as_arr)
            # df = df.drop(*features)
            # TODO: uncomment code above
            return df

        return raw_fit_data.transform(rows_to_features_and_labels)

    def split_data_to_fit_and_unused(self, joined_data: Types.RawData) -> Tuple[Types.RawData, Types.RawData]:
        # TODO: check that works properly
        user_inters_count = joined_data.groupBy("user_actor_id").count()
        # user_inters_count = joined_data.reduceByKey(lambda x, y: x + 1)
        fit_users = user_inters_count.filter(user_inters_count["count"] >= self.n_samples_threshold)["user_actor_id"]
        # fit_users = user_inters_count.filter(x)["user_actor_id"]

        fit_row_mask = joined_data["user_actor_id"].isin(fit_users)
        fit_inters = joined_data.filter(fit_row_mask)
        new_unused_inters = joined_data.filter(fit_row_mask == 0)  # debug == 0 works properly
        return fit_inters, new_unused_inters

    def save_unused(self, unused_data: Types.RawData):
        self.unused_interacts = unused_data
