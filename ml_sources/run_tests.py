import unittest
import shutil
from tests.helpers import tests_config

config = tests_config.TestsConfig()

loader = unittest.TestLoader()
start_dir = './'
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
runner.run(suite)

# deleting dirrectory with saved results
save_dir = config.save_dir
try:
    shutil.rmtree(save_dir)
except OSError as e:
    print("Error while deleting save dir: %s - %s." % (e.filename, e.strerror))


