from data.obtain import SimpleObtainer, SparkObtainer
import os
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType


class ObtainerCreator:
    def __init__(self, global_params: dict, common_params):
        self.params = global_params["obtainer_params"]
        self.paths_params = common_params["paths"]
        self.mode = common_params["mode"]
        self.log_level = common_params["log_level"]
        self.common_params = common_params
        self.columns_names = self.params["interacts_columns_names"]

        if self.mode not in ["spark", "local"]:
            raise ValueError(self.mode)

    def get_recommend_obtainer(self, dir_name: str):
        schema = StructType([
            StructField(self.columns_names[0], IntegerType(), nullable=True),
        ])
        stream_path = os.path.join(self.paths_params["base_path"], self.paths_params["worker_dir"], dir_name)
        return self._get_with_schema_and_path(schema, stream_path)

    def get_fit_obtainer(self):
        schema = StructType([
            StructField(self.columns_names[0], IntegerType(), nullable=True),
            StructField(self.columns_names[1], IntegerType(), nullable=True),
            StructField(self.columns_names[2], BooleanType(), nullable=True),
        ])
        stream_path = os.path.join(self.paths_params["base_path"], self.params["fit_stream"]["relative_path"])
        return self._get_with_schema_and_path(schema, stream_path)

    def _get_with_schema_and_path(self, schema: StructType, stream_path: str):
        if self.mode == "local":
            return SimpleObtainer()

        elif self.mode == "spark":
            stream_type = self.params["fit_stream"]["type"]
            app_name = self.common_params["app_name"]
            return SparkObtainer(stream_type, stream_path, app_name, schema, self.log_level)
