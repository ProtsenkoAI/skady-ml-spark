from business_rules.creators import RecommendsPostprocessorCreator
from model.recommend.local_recommender import LocalRecommender
from model.recommend.spark_recommender import SparkRecommender
from model.expose.recommender import Recommender
from data import ObtainerCreator
from model.creators.saver_creator import SaverCreator
from data.building_loaders import UserItemsLoaderBuilder  # TODO: make creator for loader builder (now violates layering)
import os


class RecommenderCreator:
    def __init__(self, config, common_params):
        self.config = config
        self.common_params = common_params
        self.params = config["recommender_params"]
        self.paths_info = common_params["paths"]

        self.batch_size = self.params["batch_size"]
        self.mode = common_params["mode"]

    def get(self) -> Recommender:
        load_builder = UserItemsLoaderBuilder(batch_size=self.batch_size)
        if self.mode == "local":
            return LocalRecommender(load_builder)
        elif self.mode == "spark":
            worker_dir = os.path.join(self.paths_info["base_path"], self.paths_info["worker_dir"])
            model_path = os.path.join(worker_dir, self.paths_info["model_checkpoint_name"])
            recommend_out_path = os.path.join(worker_dir, self.params["recommend_output_dir_name"])
            recommend_input_path = os.path.join(worker_dir, self.params["recommend_input_dir_name"])

            spark_saver = SaverCreator(self.config, self.common_params).get()

            os.makedirs(recommend_input_path, exist_ok=True)
            os.makedirs(recommend_out_path, exist_ok=True)

            obtainer = ObtainerCreator(self.config, self.common_params
                                       ).get_recommend_obtainer(self.params["recommend_input_dir_name"])
            data_manager = obtainer.get_data()
            recomm_postproc = RecommendsPostprocessorCreator(self.config, self.common_params).get()

            return SparkRecommender(data_manager, spark_saver, load_builder, recomm_postproc,
                                    recommend_input_path, recommend_out_path)
