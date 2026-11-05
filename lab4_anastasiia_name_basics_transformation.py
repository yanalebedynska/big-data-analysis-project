import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.window import Window

from lab3_anastasiia_name_basics_extraction import load_name_basics


def read_imdb(spark, path, schema):
    return (
        spark.read
        .schema(schema)
        .option("header", True)
        .option("sep", "\t")
        .option("nullValue", "\\N")
        .csv(str(path))
    )


def separate_professions(names):
    return (
        names
        .withColumn("profession", F.explode(F.split(F.col("primaryProfession"), ",")))
        .withColumn("profession", F.trim(F.col("profession")))
        .filter(F.col("profession") != "")
    )


def separate_directors(crew):
    return (
        crew
        .filter(F.col("directors").isNotNull())
        .select("tconst", F.explode(F.split(F.col("directors"), ",")).alias("nconst"))
        .filter(F.col("nconst") != "")
        .dropDuplicates(["tconst", "nconst"])
    )


def main():
    dataset_dir = Path(__file__).resolve().parent.parent / "datasets" / "imdb"
    filenames = ["name.basics.tsv.gz", "title.basics.tsv.gz", "title.crew.tsv.gz", "title.ratings.tsv.gz"]
    for filename in filenames:
        if not (dataset_dir / filename).is_file():
            raise FileNotFoundError(f"File not found: {dataset_dir / filename}")

    spark = (
        SparkSession.builder
        .appName("IMDbNamesTransformation")
        .master("local[1]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        names = load_name_basics(spark, dataset_dir / "name.basics.tsv.gz")

        titles_schema = StructType([
            StructField("tconst", StringType(), True),
            StructField("titleType", StringType(), True),
            StructField("primaryTitle", StringType(), True),
            StructField("originalTitle", StringType(), True),
            StructField("isAdult", IntegerType(), True),
            StructField("startYear", IntegerType(), True),
            StructField("endYear", IntegerType(), True),
            StructField("runtimeMinutes", IntegerType(), True),
            StructField("genres", StringType(), True)
        ])
        crew_schema = StructType([
            StructField("tconst", StringType(), True),
            StructField("directors", StringType(), True),
            StructField("writers", StringType(), True)
        ])
        ratings_schema = StructType([
            StructField("tconst", StringType(), True),
            StructField("averageRating", DoubleType(), True),
            StructField("numVotes", IntegerType(), True)
        ])

        titles = read_imdb(spark, dataset_dir / "title.basics.tsv.gz", titles_schema)
        crew = read_imdb(spark, dataset_dir / "title.crew.tsv.gz", crew_schema)
        ratings = read_imdb(spark, dataset_dir / "title.ratings.tsv.gz", ratings_schema)

        print("\nQUESTION 1: Directors born after 1990")
        q1 = (
            names
            .filter((F.col("birthYear") > 1990) & F.array_contains(F.split(F.col("primaryProfession"), ","), "director"))
            .select("nconst", "primaryName", "birthYear", "primaryProfession")
            .orderBy("birthYear", "primaryName")
        )
        q1.show(20, truncate=False)

        print("\nQUESTION 2: People who lived more than 90 years (CE dates)")
        q2 = (
            names
            .filter(
                (F.col("birthYear") > 0)
                & (F.col("deathYear") > 0)
                & (F.col("deathYear") >= F.col("birthYear"))
                & ((F.col("deathYear") - F.col("birthYear")) > 90)
            )
            .withColumn("ageYearsApprox", F.col("deathYear") - F.col("birthYear"))
            .select("nconst", "primaryName", "birthYear", "deathYear", "ageYearsApprox")
            .orderBy(F.desc("ageYearsApprox"), "primaryName")
        )
        q2.show(20, truncate=False)

        print("\nQUESTION 3: Most common professions")
        q3 = (
            separate_professions(names)
            .groupBy("profession")
            .agg(F.countDistinct("nconst").alias("peopleCount"))
            .orderBy(F.desc("peopleCount"), "profession")
        )
        q3.show(20, truncate=False)

        directors = separate_directors(crew)
        movies_since_2000 = (
            titles
            .filter((F.col("titleType") == "movie") & (F.col("startYear") >= 2000))
            .select("tconst")
        )
        print("\nQUESTION 4: Top 20 directors by movies released since 2000")
        q4_counts = (
            directors
            .join(movies_since_2000, "tconst", "inner")
            .groupBy("nconst")
            .agg(F.countDistinct("tconst").alias("movieCount"))
        )
        q4 = (
            q4_counts
            .join(names.select("nconst", "primaryName"), "nconst", "inner")
            .select("nconst", "primaryName", "movieCount")
            .orderBy(F.desc("movieCount"), "primaryName")
        )
        q4.show(20, truncate=False)

        print("\nQUESTION 5: Three earliest-born people per profession (1900-2000)")
        q5_people = (
            separate_professions(names)
            .filter(F.col("birthYear").between(1900, 2000))
            .select("profession", "nconst", "primaryName", "birthYear")
        )
        q5_window = Window.partitionBy("profession").orderBy(F.asc("birthYear"), F.asc("nconst"))
        q5 = (
            q5_people
            .withColumn("rank", F.row_number().over(q5_window))
            .filter(F.col("rank") <= 3)
            .orderBy("profession", "rank")
        )
        q5.show(60, truncate=False)

        print("\nQUESTION 6: Top three directors by average movie rating per genre (at least five rated movies)")
        movie_genres = (
            titles
            .filter((F.col("titleType") == "movie") & F.col("genres").isNotNull())
            .select("tconst", F.explode(F.split(F.col("genres"), ",")).alias("genre"))
            .filter(F.col("genre") != "")
        )
        rated_movies = (
            movie_genres
            .join(ratings.select("tconst", "averageRating"), "tconst", "inner")
            .filter(F.col("averageRating").isNotNull())
        )
        director_ratings = (
            directors
            .join(rated_movies, "tconst", "inner")
            .groupBy("genre", "nconst")
            .agg(
                F.countDistinct("tconst").alias("ratedMovieCount"),
                F.round(F.avg("averageRating"), 3).alias("meanRating")
            )
            .filter(F.col("ratedMovieCount") >= 5)
        )
        q6_window = Window.partitionBy("genre").orderBy(
            F.desc("meanRating"), F.desc("ratedMovieCount"), F.asc("nconst")
        )
        q6 = (
            director_ratings
            .withColumn("rank", F.row_number().over(q6_window))
            .filter(F.col("rank") <= 3)
            .join(names.select("nconst", "primaryName"), "nconst", "inner")
            .select("genre", "rank", "nconst", "primaryName", "ratedMovieCount", "meanRating")
            .orderBy("genre", "rank")
        )
        q6.show(60, truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
