import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("PySparkTest") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

data = [
    (1, "Anna"),
    (2, "Maria"),
    (3, "Olena")
]

columns = ["id", "name"]

df = spark.createDataFrame(data, columns)

df.show()

spark.stop()