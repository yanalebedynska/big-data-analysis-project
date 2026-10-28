import os
import sys
import csv

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, count, avg, round, row_number
from pyspark.sql.window import Window

from lab2_yana_ratings import read_ratings


spark = SparkSession.builder \
    .appName("IMDbRatingsBusinessQuestions") \
    .master("local[1]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


ratings_path = r"I:\4 курс\1 семестр\ОАВД\dataset\title.ratings.tsv.gz"
basics_path = r"I:\4 курс\1 семестр\ОАВД\dataset\title.basics.tsv.gz"

output_path = r"I:\4 курс\1 семестр\ОАВД\big-data-analysis-project\output"


ratings_df = read_ratings(spark, ratings_path)


basics_df = spark.read \
    .option("header", True) \
    .option("sep", "\t") \
    .option("nullValue", "\\N") \
    .csv(basics_path) \
    .select(
        "tconst",
        "titleType",
        "primaryTitle",
        "startYear"
    )


print("\nBUSINESS QUESTION 1")
print("Titles with rating >= 8.5 and at least 10000 votes")

question_1 = ratings_df.filter(
    (col("averageRating") >= 8.5) &
    (col("numVotes") >= 10000)
).orderBy(
    col("averageRating").desc(),
    col("numVotes").desc()
)

question_1.show(20, truncate=False)


print("\nBUSINESS QUESTION 2")
print("Top 20 titles with rating >= 7.0 by number of votes")

question_2 = ratings_df.filter(
    col("averageRating") >= 7.0
).orderBy(
    col("numVotes").desc()
).limit(20)

question_2.show(20, truncate=False)


print("\nBUSINESS QUESTION 3")
print("Number of titles in different rating categories")

question_3_data = ratings_df.filter(
    col("numVotes") >= 100
).withColumn(
    "ratingCategory",
    when(col("averageRating") >= 8.0, "High")
    .when(col("averageRating") >= 6.0, "Medium")
    .otherwise("Low")
)

question_3 = question_3_data.groupBy(
    "ratingCategory"
).agg(
    count("*").alias("objectsCount")
).orderBy(
    col("objectsCount").desc()
)

question_3.show(truncate=False)


print("\nBUSINESS QUESTION 4")
print("Average rating by title type for titles with at least 1000 votes")

question_4_data = ratings_df.join(
    basics_df,
    on="tconst",
    how="inner"
).filter(
    col("numVotes") >= 1000
)

question_4 = question_4_data.groupBy(
    "titleType"
).agg(
    round(avg("averageRating"), 2).alias("averageRating"),
    count("*").alias("objectsCount")
).orderBy(
    col("averageRating").desc()
)

question_4.show(20, truncate=False)


print("\nBUSINESS QUESTION 5")
print("Top 5 titles by number of votes within each title type")

question_5_data = ratings_df.join(
    basics_df,
    on="tconst",
    how="inner"
).filter(
    col("titleType").isNotNull()
)

window_5 = Window.partitionBy(
    "titleType"
).orderBy(
    col("numVotes").desc()
)

question_5 = question_5_data.withColumn(
    "rank",
    row_number().over(window_5)
).filter(
    col("rank") <= 5
).select(
    "titleType",
    "primaryTitle",
    "averageRating",
    "numVotes",
    "rank"
).orderBy(
    "titleType",
    "rank"
)

question_5.show(50, truncate=False)


print("\nBUSINESS QUESTION 6")
print("Top 3 movies by rating for each year with at least 50000 votes")

question_6_data = ratings_df.join(
    basics_df,
    on="tconst",
    how="inner"
).filter(
    (col("titleType") == "movie") &
    (col("startYear").isNotNull()) &
    (col("numVotes") >= 50000)
)

window_6 = Window.partitionBy(
    "startYear"
).orderBy(
    col("averageRating").desc(),
    col("numVotes").desc()
)

question_6 = question_6_data.withColumn(
    "rank",
    row_number().over(window_6)
).filter(
    col("rank") <= 3
).select(
    "startYear",
    "primaryTitle",
    "averageRating",
    "numVotes",
    "rank"
).orderBy(
    col("startYear").desc(),
    col("rank")
)

question_6.show(30, truncate=False)


os.makedirs(output_path, exist_ok=True)


def save_to_csv(df, file_name):
    file_path = os.path.join(output_path, file_name)

    with open(file_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        writer.writerow(df.columns)

        for row in df.toLocalIterator():
            writer.writerow([row[column] for column in df.columns])


print("\nSAVING RESULTS TO CSV")

save_to_csv(question_1, "question_1.csv")
print("question_1.csv saved")

save_to_csv(question_2, "question_2.csv")
print("question_2.csv saved")

save_to_csv(question_3, "question_3.csv")
print("question_3.csv saved")

save_to_csv(question_4, "question_4.csv")
print("question_4.csv saved")

save_to_csv(question_5, "question_5.csv")
print("question_5.csv saved")

save_to_csv(question_6, "question_6.csv")
print("question_6.csv saved")


print("\nALL BUSINESS QUESTION RESULTS SAVED TO CSV")


spark.stop()