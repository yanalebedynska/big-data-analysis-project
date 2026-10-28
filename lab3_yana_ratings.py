import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from lab2_yana_ratings import read_ratings

spark = SparkSession.builder \
    .appName("IMDbRatingsTransformation") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

file_path = r"I:\4 курс\1 семестр\ОАВД\dataset\title.ratings.tsv.gz"

ratings_df = read_ratings(spark, file_path)

print("GENERAL INFORMATION")
print("Number of rows:", ratings_df.count())
print("Number of columns:", len(ratings_df.columns))
print("Columns:", ratings_df.columns)

print("SCHEMA")
ratings_df.printSchema()

print("STATISTICS")
ratings_df.select("averageRating", "numVotes").describe().show()

spark.stop()
