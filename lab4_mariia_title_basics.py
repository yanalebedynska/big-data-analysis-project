from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from extract_title_basics import extract_title_basics


def by_genre(df):
    return (
        df.filter(F.col("genres").isNotNull())
        .withColumn("genre", F.explode(F.split("genres", ",")))
    )


def business_queries(movies, ratings):
    q1 = (
        movies.filter(
            F.col("startYear").between(2020, 2025)
            & F.col("runtimeMinutes").between(60, 120)
        )
        .select(
            "tconst", "primaryTitle", "startYear",
            "runtimeMinutes", "genres"
        )
        .orderBy(F.desc("startYear"), "runtimeMinutes", "tconst")
        .limit(20)
    )

    recent = movies.filter(F.col("startYear").between(2016, 2025))

    q2 = (
        by_genre(recent)
        .groupBy("genre")
        .agg(F.countDistinct("tconst").alias("movie_count"))
        .orderBy(F.desc("movie_count"), "genre")
    )

    yearly = (
        movies.filter(F.col("startYear").between(2015, 2025))
        .groupBy("startYear")
        .agg(F.countDistinct("tconst").alias("movie_count"))
    )

    year_window = Window.orderBy("startYear")

    q3 = (
        yearly
        .withColumn(
            "previous_year",
            F.lag("startYear").over(year_window)
        )
        .withColumn(
            "previous_count",
            F.lag("movie_count").over(year_window)
        )
        .withColumn(
            "change_pct",
            F.when(
                (F.col("startYear") == F.col("previous_year") + 1)
                & (F.col("previous_count") > 0),
                F.round(
                    (F.col("movie_count") - F.col("previous_count"))
                    / F.col("previous_count") * 100,
                    2
                )
            )
        )
        .filter(F.col("startYear").between(2016, 2025))
        .select(
            "startYear", "movie_count",
            "previous_count", "change_pct"
        )
        .orderBy("startYear")
    )

    q4 = (
        movies.filter(F.col("startYear").between(1960, 1999))
        .join(ratings, on="tconst", how="inner")
        .filter(
            (F.col("averageRating") >= 8.0)
            & (F.col("numVotes") >= 10000)
        )
        .select(
            "tconst", "primaryTitle", "startYear",
            "averageRating", "numVotes"
        )
        .orderBy(
            F.desc("averageRating"), F.desc("numVotes"), "tconst"
        )
        .limit(20)
    )

    rated_recent = (
        recent.join(ratings, on="tconst", how="inner")
        .filter(
            (F.col("averageRating") >= 7.0)
            & (F.col("numVotes") >= 10000)
        )
    )

    q5 = (
        by_genre(rated_recent)
        .groupBy("genre")
        .agg(
            F.countDistinct("tconst").alias("movie_count"),
            F.round(
                F.avg("averageRating"), 2
            ).alias("mean_rating")
        )
        .orderBy(F.desc("movie_count"), "genre")
    )

    short_candidates = (
        movies.filter(
            F.col("startYear").between(2020, 2025)
            & F.col("runtimeMinutes").between(60, 100)
        )
        .join(ratings, on="tconst", how="inner")
        .filter(
            (F.col("averageRating") >= 7.0)
            & (F.col("numVotes") >= 10000)
        )
    )

    genre_window = Window.partitionBy("genre").orderBy(
        F.desc("averageRating"), F.desc("numVotes"), "tconst"
    )

    q6 = (
        by_genre(short_candidates)
        .filter(
            F.col("genre").isin(
                "Action", "Animation", "Comedy", "Drama"
            )
        )
        .withColumn("position", F.row_number().over(genre_window))
        .filter(F.col("position") <= 3)
        .select(
            "genre", "position", "primaryTitle", "startYear",
            "runtimeMinutes", "averageRating", "numVotes"
        )
        .orderBy("genre", "position")
    )

    return [
        ("Кандидати для двогодинного сеансу", q1),
        ("Кількість фільмів за жанрами", q2),
        ("Динаміка кількості фільмів за роками", q3),
        ("Фільми для ретроспективних показів", q4),
        ("Добре оцінені фільми за жанрами", q5),
        ("Топ-3 фільмів у кожному вибраному жанрі", q6),
    ]


def main():
    data_dir = Path(r"C:\LPNU\4 course\OABD\dataset")

    for name in ("title.basics.tsv.gz", "title.ratings.tsv.gz"):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"Файл не знайдено: {data_dir / name}")

    spark = (
        SparkSession.builder
        .appName("Lab4MariiaTitleBasics")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        basics = extract_title_basics(
            spark, (data_dir / "title.basics.tsv.gz").as_posix()
        )

        ratings = (
            spark.read
            .schema(
                "tconst STRING, averageRating DOUBLE, numVotes INT"
            )
            .option("header", True)
            .option("sep", "\t")
            .option("nullValue", "\\N")
            .option("mode", "FAILFAST")
            .csv((data_dir / "title.ratings.tsv.gz").as_posix())
            .cache()
        )

        movies = (
            basics.filter(
                (F.col("titleType") == "movie")
                & (F.col("isAdult") == 0)
            )
            .select(
                "tconst", "primaryTitle", "startYear",
                "runtimeMinutes", "genres"
            )
            .cache()
        )

        for number, (title, result) in enumerate(
            business_queries(movies, ratings), start=1
        ):
            print(f"\n=== ПИТАННЯ {number}: {title} ===")
            result.show(30, truncate=30)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()