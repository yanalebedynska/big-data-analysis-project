import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from lab2_yana_ratings import read_ratings

spark = SparkSession.builder \
    .appName("IMDbRatingsExtraction") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

file_path = r"I:\4 курс\1 семестр\ОАВД\dataset\title.ratings.tsv.gz"

ratings_df = read_ratings(spark, file_path)

print("SCHEMA")
ratings_df.printSchema()

print("FIRST 10 ROWS")
ratings_df.show(10, truncate=False)

print("NUMBER OF ROWS")
print(ratings_df.count())

spark.stop()
