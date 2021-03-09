"""
1. Find path to spark-submit script
2. Make .zip from dependencies to transmit it to workers on cluster
3. Make a package with sources and configuration to transmit it
4. Run submit
"""
import os
import subprocess

spark_home = os.environ["SPARK_HOME"]
submit_file_path = os.path.join(spark_home, "bin", "spark-submit")
print(submit_file_path)

subprocess.check_output([submit_file_path, "--master", ""])
