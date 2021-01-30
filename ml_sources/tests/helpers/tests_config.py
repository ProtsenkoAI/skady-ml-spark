import pathlib


class TestsConfig:
    def __init__(self):
        self.data_path = (pathlib.Path(__file__).parent.parent.parent.parent / "data").absolute()
        self.interacts_path = str(self.data_path / "rating.csv")