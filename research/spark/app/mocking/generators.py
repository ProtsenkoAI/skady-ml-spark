import os
from random import randint
from typing import List
import pandas as pd
import numpy as np

# Sample = str
# Samples = List[Sample]
Samples = pd.DataFrame


class CsvDirGenerator:
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
        df = pd.DataFrame(np.random.randint(0, self.nusers, (n_lines_created, 3)))
        df[0] = df[0].astype(int)
        df[1] = df[1].astype(int)
        df[2] = df[2].astype(bool)
        return df

    def _push(self, samples: Samples):
        # random_file_name = os.path.join(self.dir_path, f"{self.nfiles}.txt")
        random_file_name = os.path.join(self.dir_path, f"{self.nfiles}.csv")
        # with open(random_file_name, "w") as f:
            # samples_with_newlines = "\n".join(samples)
            # samples_with_newlines = "bebebe\nsbababa"
            # f.write(samples_with_newlines)
        samples.to_csv(random_file_name, index=False, header=False)
        self.nfiles += 1
