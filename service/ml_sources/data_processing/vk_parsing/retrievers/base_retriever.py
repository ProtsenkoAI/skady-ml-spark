class Retriever:
    def __init__(self):
        self.logs = ""

    def get(self, obtainer):
        raise NotImplementedError

    def reset_session(self, new_session):
        raise NotImplementedError

    # def continue_in_new_session(self, new_session, obtainer):
    #     self.reset_session(new_session)
    #     return self.get(obtainer)

    def get_logs(self):
        return self.logs

    def log(self, string):
        new_line = "\n"
        self.logs += string + new_line
