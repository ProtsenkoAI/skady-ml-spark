from random import randint


class Validator:
    def __init__(self, assistant):
        self.assistant = assistant

    def evaluate(self):
        return randint(0, 1)
