import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from extract_title_basics import extract_title_basics


def main():
    file_path = Path(r"C:\LPNU\4 course\OABD\dataset\title.basics.tsv.gz")

    if not file_path.is_file():
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")

    spark = (
        SparkSession.builder
        .appName("Lab3TitleBasics")
        .master("local[1]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = extract_title_basics(spark, file_path.as_posix())

        print("\n=== СХЕМА DATAFRAME ===")
        df.printSchema()

        print("\n=== ПЕРШІ 10 ЗАПИСІВ ===")
        df.show(10, truncate=35)

        print("\n=== РОЗМІР DATAFRAME ===")
        print(f"Кількість стовпців: {len(df.columns)}")
        print(f"Кількість рядків: {df.count()}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()