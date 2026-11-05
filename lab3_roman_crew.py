import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, size, when, sum, explode, countDistinct
from lab2_roman_crew import read_crew

spark = SparkSession.builder \
    .appName("IMDbCrewTransformation") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

file_path = "/Users/roman/Documents/оавд/dataset/title.crew.tsv.gz"

crew_df = read_crew(spark, file_path)

print("GENERAL INFORMATION")
print("Number of rows:", crew_df.count())
print("Number of columns:", len(crew_df.columns))
print("Columns:", crew_df.columns)
print("Unique titles:", crew_df.select("tconst").distinct().count())

print("SCHEMA")
crew_df.printSchema()

print("MISSING VALUES")
crew_df.select(
    sum(when(col("directors").isNull(), 1).otherwise(0)).alias("directorsNull"),
    sum(when(col("writers").isNull(), 1).otherwise(0)).alias("writersNull"),
    sum(when(col("directors").isNull() & col("writers").isNull(), 1).otherwise(0)).alias("bothNull")
).show()

print("UNIQUE PEOPLE")
print("Unique directors:", crew_df.select(explode(split(col("directors"), ",")).alias("id")).agg(countDistinct("id")).first()[0])
print("Unique writers:", crew_df.select(explode(split(col("writers"), ",")).alias("id")).agg(countDistinct("id")).first()[0])

crew_counts_df = crew_df.withColumn(
    "directorsCount",
    when(col("directors").isNull(), 0).otherwise(size(split(col("directors"), ",")))
).withColumn(
    "writersCount",
    when(col("writers").isNull(), 0).otherwise(size(split(col("writers"), ",")))
)

print("STATISTICS")
crew_counts_df.select("directorsCount", "writersCount").describe().show()

print("MEDIAN AND 99th PERCENTILE")
print("directorsCount:", crew_counts_df.approxQuantile("directorsCount", [0.5, 0.99], 0.001))
print("writersCount:", crew_counts_df.approxQuantile("writersCount", [0.5, 0.99], 0.001))

spark.stop()
