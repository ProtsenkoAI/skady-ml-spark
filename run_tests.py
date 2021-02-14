import unittest
import shutil
import os

from helpers import tests_config


def run_tests(start_dir):

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)


def delete_saved_results():
    save_dir = config.save_dir
    try:
        shutil.rmtree(save_dir, ignore_errors=True)
    except OSError as e:
        print("Error while deleting save dir: %s - %s." % (e.filename, e.strerror))


config = tests_config.TestsConfig()
# starting from dir with this file
this_file_dir = os.path.dirname(os.path.abspath(__file__))
run_tests(start_dir=this_file_dir)

delete_saved_results()
