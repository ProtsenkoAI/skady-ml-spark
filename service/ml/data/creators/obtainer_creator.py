from data.obtain import SimpleObtainer, SparkObtainer
import os
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType


class ObtainerCreator:
    # TODO: refactor so can leave obtainer creators for old tasks unchanged when adding new obtainers (use factory)
    def __init__(self, global_params: dict, common_params):
        self.params = global_params["obtainer_params"]
        self.paths_params = common_params["paths"]
        self.mode = common_params["mode"]
        self.log_level = common_params["log_level"]
        self.common_params = common_params
        self.columns_names = self.params["interacts_columns_names"]
        self.worker_dir_path = os.path.join(self.paths_params["base_path"], self.paths_params["worker_dir"])

        if self.mode not in ["spark", "local"]:
            raise ValueError(self.mode)

    def get_recommend_obtainer(self, dir_name: str):
        schema = StructType([
            StructField(self.columns_names[0], IntegerType(), nullable=True),
        ])
        stream_path = os.path.join(self.worker_dir_path, dir_name)
        return self._get_with_schema_and_path(schema, stream_path)

    def get_fit_obtainer(self):
        schema = StructType([
            StructField(self.columns_names[0], IntegerType(), nullable=True),
            StructField(self.columns_names[1], IntegerType(), nullable=True),
            StructField(self.columns_names[2], BooleanType(), nullable=True),
        ])
        stream_path = os.path.join(self.paths_params["base_path"], self.params["fit_stream"]["relative_path"])
        return self._get_with_schema_and_path(schema, stream_path)

    def get_vk_embed_tasks_obtainer(self):
        schema = StructType([
            StructField("task", StringType(), nullable=True),
            StructField("user_id", IntegerType(), nullable=True),
        ])
        path = os.path.join(self.worker_dir_path, self.paths_params["vk_obtainer_tasks_dir"])
        print("tasks obtainer will listen the tasks on", path)
        return self._get_with_schema_and_path(schema, path)

    def _get_with_schema_and_path(self, schema: StructType, stream_path: str):
        if self.mode == "local":
            return SimpleObtainer()

        elif self.mode == "spark":
            stream_type = self.params["fit_stream"]["type"]
            app_name = self.common_params["app_name"]
            return SparkObtainer(stream_type, stream_path, app_name, schema, self.log_level)
