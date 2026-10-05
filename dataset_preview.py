import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("IMDbDatasetPreview") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

dataset_path = r"I:\4 курс\1 семестр\ОАВД\dataset"

title_basics = spark.read \
    .option("header", True) \
    .option("sep", "\t") \
    .option("nullValue", "\\N") \
    .csv(dataset_path + r"\title.basics.tsv.gz")

print("TITLE BASICS")
print("Columns:")
print(title_basics.columns)

title_basics.show(10, truncate=False)

title_ratings = spark.read \
    .option("header", True) \
    .option("sep", "\t") \
    .option("nullValue", "\\N") \
    .csv(dataset_path + r"\title.ratings.tsv.gz")

print("TITLE RATINGS")
print("Columns:")
print(title_ratings.columns)

title_ratings.show(10, truncate=False)

title_crew = spark.read \
    .option("header", True) \
    .option("sep", "\t") \
    .option("nullValue", "\\N") \
    .csv(dataset_path + r"\title.crew.tsv.gz")

print("TITLE CREW")
print("Columns:")
print(title_crew.columns)

title_crew.show(10, truncate=False)

name_basics = spark.read \
    .option("header", True) \
    .option("sep", "\t") \
    .option("nullValue", "\\N") \
    .csv(dataset_path + r"\name.basics.tsv.gz")

print("NAME BASICS")
print("Columns:")
print(name_basics.columns)

name_basics.show(10, truncate=False)

spark.stop()