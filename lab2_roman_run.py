import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from lab2_roman_crew import read_crew

spark = SparkSession.builder \
    .appName("IMDbCrewExtraction") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

file_path = "/Users/roman/Documents/оавд/dataset/title.crew.tsv.gz"

crew_df = read_crew(spark, file_path)

print("SCHEMA")
crew_df.printSchema()

print("FIRST 10 ROWS")
crew_df.show(10, truncate=False)

print("NUMBER OF ROWS")
print(crew_df.count())

spark.stop()
