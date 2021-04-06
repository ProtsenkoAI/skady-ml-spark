import uuid
import pandas as pd
import os
from data.expose import StreamingDataManager


class VkUserEmbeddingObtainer:
    def __init__(self, inp_tasks_manager: StreamingDataManager, vk_tasks_dir: str):
        self.inp_tasks_manager = inp_tasks_manager
        self.vk_tasks_dir = vk_tasks_dir
        self._started = False

    def add_user(self, user):
        self._start_spark_if_needed()
        self._write_to_spark_tasks_dir(user, mode="add")

    def delete_user(self, user):
        self._start_spark_if_needed()
        self._write_to_spark_tasks_dir(user, mode="delete")

    def _start_spark_if_needed(self):
        if not self._started:
            print("starting spark")
            self._start_spark()
        self._started = True

    def _start_spark(self):
        self.inp_tasks_manager.apply_to_each_row(self._add_or_delete_user_embeds_write_changes)

    @staticmethod
    def _add_or_delete_user_embeds_write_changes(row: pd.DataFrame):
        print("called from spark")
        mode = row["task"]
        user_id = row["user_id"]

        if mode not in ["add", "delete"]:
            raise ValueError("mode in embeddings obtainer: ", mode)
        if mode == "add":
            print("add in spark", user_id)
        elif mode == "delete":
            print("delete in spark", user_id)

    def _write_to_spark_tasks_dir(self, user, mode):
        some_name = f"{uuid.uuid4()}.csv"
        path_to_task_file = os.path.join(self.vk_tasks_dir, some_name)
        print("dumping the task to", path_to_task_file)
        df = pd.DataFrame({"task": mode, "user_id": [user]})
        df.to_csv(path_to_task_file, index=False, header=False)
