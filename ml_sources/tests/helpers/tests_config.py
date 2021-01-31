import pathlib


class TestsConfig:
    def __init__(self):
        self.config_path = pathlib.Path(__file__)
        self.project_root = self.config_path.parents[3]
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = "cpu"
        self.user_colname = "user_id"
        self.item_colname = "anime_id"

        self.data_path = (self.project_root / "data").absolute()
        self.interacts_path = str(self.data_path / "rating.csv")
        self.save_dir = str(self.project_root / "tmp_results")
