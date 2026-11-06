import os
import sys
import csv

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, size, when, explode, arrays_overlap, count, avg, sum,
    round, row_number, floor
)
from pyspark.sql.window import Window

from lab2_roman_crew import read_crew
from lab2_yana_ratings import read_ratings


spark = SparkSession.builder \
    .appName("IMDbCrewBusinessQuestions") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


dataset_path = "/Users/roman/Documents/оавд/dataset"

crew_path = os.path.join(dataset_path, "title.crew.tsv.gz")
ratings_path = os.path.join(dataset_path, "title.ratings.tsv.gz")
basics_path = os.path.join(dataset_path, "title.basics.tsv.gz")
names_path = os.path.join(dataset_path, "name.basics.tsv.gz")

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "roman")


crew_df = read_crew(spark, crew_path) \
    .withColumn("directorsList", split(col("directors"), ",")) \
    .withColumn("writersList", split(col("writers"), ","))

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
        col("startYear").cast("int").alias("startYear")
    )


names_df = spark.read \
    .option("header", True) \
    .option("sep", "\t") \
    .option("nullValue", "\\N") \
    .csv(names_path) \
    .select(
        col("nconst").alias("director"),
        col("primaryName").alias("directorName")
    )


# One row per title and director
directors_df = crew_df.filter(
    col("directors").isNotNull()
).select(
    "tconst",
    explode("directorsList").alias("director")
)

movie_directors_df = directors_df.join(
    basics_df.filter(col("titleType") == "movie"),
    on="tconst",
    how="inner"
)

rated_movie_directors_df = movie_directors_df.join(
    ratings_df,
    on="tconst",
    how="inner"
)


print("\nBUSINESS QUESTION 1")
print("Titles where at least one director is also a writer")

question_1_data = crew_df.filter(
    col("directors").isNotNull() &
    col("writers").isNotNull() &
    arrays_overlap(col("directorsList"), col("writersList"))
)

print("Number of titles:", question_1_data.count())
print("Titles with both directors and writers:", crew_df.filter(
    col("directors").isNotNull() & col("writers").isNotNull()
).count())

question_1 = question_1_data.join(
    basics_df,
    on="tconst",
    how="inner"
).join(
    ratings_df,
    on="tconst",
    how="inner"
).filter(
    col("titleType") == "movie"
).select(
    "tconst",
    "primaryTitle",
    "startYear",
    "directors",
    "writers",
    "numVotes"
).orderBy(
    col("numVotes").desc()
).limit(20)

question_1.show(20, truncate=False)


print("\nBUSINESS QUESTION 2")
print("Number of titles by number of directors")

question_2 = crew_df.withColumn(
    "directorsGroup",
    when(col("directors").isNull(), "0")
    .when(size(col("directorsList")) == 1, "1")
    .when(size(col("directorsList")) == 2, "2")
    .otherwise("3+")
).groupBy(
    "directorsGroup"
).agg(
    count("*").alias("objectsCount")
).withColumn(
    "percent",
    round(col("objectsCount") * 100 / crew_df.count(), 2)
).orderBy(
    "directorsGroup"
)

question_2.show(truncate=False)


print("\nBUSINESS QUESTION 3")
print("Top 20 directors by number of movies")

question_3 = movie_directors_df.groupBy(
    "director"
).agg(
    count("*").alias("moviesCount")
).join(
    names_df,
    on="director",
    how="left"
).select(
    "director",
    "directorName",
    "moviesCount"
).orderBy(
    col("moviesCount").desc()
).limit(20)

question_3.show(20, truncate=False)


print("\nBUSINESS QUESTION 4")
print("Top 20 directors by average movie rating (at least 5 movies with 10000+ votes)")

question_4 = rated_movie_directors_df.filter(
    col("numVotes") >= 10000
).groupBy(
    "director"
).agg(
    count("*").alias("moviesCount"),
    round(avg("averageRating"), 2).alias("avgRating"),
    sum("numVotes").alias("totalVotes")
).filter(
    col("moviesCount") >= 5
).join(
    names_df,
    on="director",
    how="left"
).select(
    "directorName",
    "moviesCount",
    "avgRating",
    "totalVotes"
).orderBy(
    col("avgRating").desc(),
    col("totalVotes").desc()
).limit(20)

question_4.show(20, truncate=False)


print("\nBUSINESS QUESTION 5")
print("Top 3 directors by number of movies (1000+ votes) in each decade")

question_5_data = rated_movie_directors_df.filter(
    (col("numVotes") >= 1000) &
    (col("startYear") >= 1960) &
    (col("startYear") < 2030)
).withColumn(
    "decade",
    (floor(col("startYear") / 10) * 10).cast("int")
).groupBy(
    "decade",
    "director"
).agg(
    count("*").alias("moviesCount"),
    round(avg("averageRating"), 2).alias("avgRating")
)

window_5 = Window.partitionBy(
    "decade"
).orderBy(
    col("moviesCount").desc(),
    col("avgRating").desc()
)

question_5 = question_5_data.withColumn(
    "rank",
    row_number().over(window_5)
).filter(
    col("rank") <= 3
).join(
    names_df,
    on="director",
    how="left"
).select(
    "decade",
    "directorName",
    "moviesCount",
    "avgRating",
    "rank"
).orderBy(
    "decade",
    "rank"
)

question_5.show(30, truncate=False)


print("\nBUSINESS QUESTION 6")
print("Top 3 movies of the 10 directors with the most total votes")

top_directors = rated_movie_directors_df.groupBy(
    "director"
).agg(
    sum("numVotes").alias("totalVotes")
).orderBy(
    col("totalVotes").desc()
).limit(10)

window_6 = Window.partitionBy(
    "director"
).orderBy(
    col("averageRating").desc(),
    col("numVotes").desc()
)

question_6 = rated_movie_directors_df.join(
    top_directors,
    on="director",
    how="inner"
).filter(
    col("numVotes") >= 10000
).withColumn(
    "rank",
    row_number().over(window_6)
).filter(
    col("rank") <= 3
).join(
    names_df,
    on="director",
    how="left"
).select(
    "directorName",
    "totalVotes",
    "primaryTitle",
    "startYear",
    "averageRating",
    "numVotes",
    "rank"
).orderBy(
    col("totalVotes").desc(),
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
