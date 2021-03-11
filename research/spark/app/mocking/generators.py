import os
from random import randint
from typing import List

Sample = str
Samples = List[Sample]


class TextDirGenerator:
    def __init__(self, dir_path, max_nsamples_per_gen=20, nusers=1000):
        self.dir_path = dir_path
        self.max_nsamples_per_gen = max_nsamples_per_gen
        self.nusers = nusers

        self.nfiles = 0
        os.makedirs(dir_path, exist_ok=True)

    def create_push(self):
        new_data = self._create()
        self._push(new_data)

    def _create(self) -> Samples:
        n_lines_created = randint(1, self.max_nsamples_per_gen)
        lines = [self._create_line() for _ in range(n_lines_created)]
        return lines

    def _create_line(self):
        user_actor = randint(1, self.nusers)
        user_target = randint(1, self.nusers)
        result = randint(0, 1)
        return f"{user_actor} {user_target} {result}"

    def _push(self, samples: Samples):
        random_file_name = os.path.join(self.dir_path, f"{self.nfiles}.txt")
        with open(random_file_name, "w") as f:
            samples_with_newlines = "\n".join(samples)
            f.write(samples_with_newlines)
        self.nfiles += 1
