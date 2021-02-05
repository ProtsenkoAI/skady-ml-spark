class Trainer:
    def __init__(self, assistant):
        self.assistant = assistant

    def get_dataset_len(self):
        return 10

    def fit(self, nsteps=None, nepochs=None):
        ...
