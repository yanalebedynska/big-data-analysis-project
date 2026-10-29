
import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
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

    df = (
        spark.read
        .schema(schema)
        .option("header", True)
        .option("sep", "\t")
        .option("nullValue", "\\N")
        .csv(str(file_path))
    )

    return df


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

    finally:
        spark.stop()
