# TODO: at the moment can recommend deleted users, need to prevent it somehow (maybe in filtering)

from business_rules.expose import RecommendsPostprocessor
from .local_recommender import LocalRecommender
from ..expose.recommender import Recommender
from data import StreamingDataManager
from .. import Saver
from typing import List
import pandas as pd
import uuid
import os
import time


class SparkRecommender(Recommender):
    # TODO: maybe should move recommends_postprocessor to spark side?
    def __init__(self, inp_data_manager: StreamingDataManager,
                 saver: Saver, users_items_loader_builder,
                 recommends_postprocessor: RecommendsPostprocessor,
                 recommend_input_path, recommend_out_path, recommend_time_limit=100):
        self.loader_builder = users_items_loader_builder
        self.inp_data_manager = inp_data_manager
        self.saver = saver
        self.recommends_postprocessor = recommends_postprocessor
        self._started = False
        self.recommend_time_limit = recommend_time_limit
        self.recommend_input_path = recommend_input_path
        self.recommend_out_path = recommend_out_path

    def get_recommends(self, user):
        if not self._started:
            self._start_spark()
        self._send_user_to_stream(user)
        self._wait_for_recommend(user)
        raw_recommends = self._take_recommend(user)
        postprocessed_recommends = self.recommends_postprocessor.process(user, raw_recommends)
        return postprocessed_recommends

    def _start_spark(self):
        self.inp_data_manager.apply_to_each_row(self._get_recommends_push_results)

    def _get_recommends_push_results(self, batch: pd.DataFrame):
        user = batch.values[0][0]  # TODO: refactor getting user from dataframe
        packaged_model = self.saver.load_packaged_model()
        print("available items in model in recommends", packaged_model.manager.processor.get_all_items())
        local_recommender = LocalRecommender(self.loader_builder, model_manager=packaged_model.manager)
        recommends = local_recommender.get_recommends(user)
        self._push_recommends(user, recommends)

    def _push_recommends(self, user: List[int], recommends):
        file_path = os.path.join(self.recommend_out_path, f"{user}.txt")
        if os.path.isfile(file_path):
            raise ValueError("recommends for this user already exist")
        with open(file_path, "w") as f:
            recommends_str = " ".join([str(user) for user in recommends])
            f.write(recommends_str)

    def _send_user_to_stream(self, user: int):
        file_path = os.path.join(self.recommend_input_path, str(uuid.uuid4()) + ".csv")
        df = pd.DataFrame(data=[user])
        df.to_csv(file_path, index=False)

    def _wait_for_recommend(self, user):
        file_path = os.path.join(self.recommend_out_path, f"{user}.txt")

        start = time.time()
        end_time = start + self.recommend_time_limit
        while time.time() < end_time:
            if os.path.exists(file_path):
                break
            else:
                time.sleep(0.05)

    def _take_recommend(self, user) -> List:
        file_path = os.path.join(self.recommend_out_path, f"{user}.txt")
        with open(file_path) as f:
            recommends_strings = f.read().split(" ")
            recommends = [int(user_id) for user_id in recommends_strings]
        os.remove(file_path)
        return recommends
