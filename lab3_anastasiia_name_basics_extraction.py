import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


def load_name_basics(spark, file_path):
    schema = StructType([
        StructField("nconst", StringType(), True),
        StructField("primaryName", StringType(), True),
        StructField("birthYear", IntegerType(), True),
        StructField("deathYear", IntegerType(), True),
        StructField("primaryProfession", StringType(), True),
        StructField("knownForTitles", StringType(), True)
    ])

    return (
        spark.read
        .schema(schema)
        .option("header", True)
        .option("sep", "\t")
        .option("nullValue", "\\N")
        .csv(str(file_path))
    )


if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("IMDbNamesExtraction")
        .master("local[1]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        dataset_path = (
            Path(__file__).resolve().parent.parent
            / "datasets"
            / "imdb"
            / "name.basics.tsv.gz"
        )

        if not dataset_path.is_file():
            raise FileNotFoundError(f"File not found: {dataset_path}")

        names_df = load_name_basics(spark, dataset_path)

        print("NAME BASICS SCHEMA:")
        names_df.printSchema()

        print("NAME BASICS DATA:")
        names_df.show(10, truncate=False)

        print("GENERAL INFORMATION:")
        total_rows = names_df.count()
        print("Number of rows:", total_rows)
        print("Number of columns:", len(names_df.columns))
        print("Columns:", names_df.columns)
        print("Column types:", names_df.dtypes)

        print("NUMERIC STATISTICS:")
        names_df.select("birthYear", "deathYear").describe().show(truncate=False)

        print("MISSING VALUES:")
        missing_counts = names_df.select([
            F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
            for column in names_df.columns
        ])
        missing_counts.show(truncate=False)

        print("MISSING VALUES (%):")
        if total_rows > 0:
            missing_counts.select([
                F.round(F.col(column) * 100 / F.lit(total_rows), 2).alias(column)
                for column in names_df.columns
            ]).show(truncate=False)

        print("DUPLICATE NCONST VALUES:")
        id_counts = names_df.groupBy("nconst").count()
        duplicate_ids = id_counts.filter(F.col("count") > 1)
        duplicate_ids.show(10, truncate=False)
        print("Number of repeated nconst values:", duplicate_ids.count())

        print("FULLY DUPLICATED ROWS:")
        distinct_rows = names_df.distinct().count()
        print("Number of extra identical rows:", total_rows - distinct_rows)

        print("INVALID YEAR VALUES:")
        invalid_years = names_df.filter(
            (F.col("birthYear").isNotNull() & (F.col("birthYear") <= 0))
            | (F.col("deathYear").isNotNull() & (F.col("deathYear") <= 0))
            | (
                F.col("birthYear").isNotNull()
                & F.col("deathYear").isNotNull()
                & (F.col("deathYear") < F.col("birthYear"))
            )
        )
        print(
            "Number of records with invalid year order or nonpositive years:",
            invalid_years.count()
        )
        invalid_years.select(
            "nconst", "primaryName", "birthYear", "deathYear"
        ).show(10, truncate=False)

    finally:
        spark.stop()
