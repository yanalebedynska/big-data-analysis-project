import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from extract_title_basics import extract_title_basics


def main():
    file_path = Path(
        r"C:\LPNU\4 course\OABD\dataset\title.basics.tsv.gz"
    )

    if not file_path.is_file():
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")

    spark = (
        SparkSession.builder
        .appName("Lab3TitleBasicsAnalysis")
        .master("local[1]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = extract_title_basics(spark, file_path.as_posix())

        numeric_columns = [
            "isAdult",
            "startYear",
            "endYear",
            "runtimeMinutes",
        ]

        total_rows = df.count()

        print("\n=== ЗАГАЛЬНА ІНФОРМАЦІЯ ===")
        print(f"Файл: {file_path.name}")
        print(f"Кількість рядків: {total_rows}")
        print(f"Кількість стовпців: {len(df.columns)}")
        print(f"Числових стовпців: {len(numeric_columns)}")
        print(
            f"Текстових стовпців: "
            f"{len(df.columns) - len(numeric_columns)}"
        )

        print("\nСтовпці та їхні типи:")
        for column_name, data_type in df.dtypes:
            print(f"{column_name}: {data_type}")

        print("\n=== СТАТИСТИКА ЧИСЛОВИХ СТОВПЦІВ ===")

        statistics = df.select(*numeric_columns).summary(
            "count",
            "mean",
            "stddev",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        )
        statistics.show(truncate=False)

        print("\n=== ПРОПУЩЕНІ ЗНАЧЕННЯ ===")

        missing_counts = df.agg(*[
            F.count(
                F.when(F.col(column_name).isNull(), 1)
            ).alias(column_name)
            for column_name in df.columns
        ]).first()

        for column_name in df.columns:
            missing = missing_counts[column_name]
            percent = missing / total_rows * 100 if total_rows else 0

            print(
                f"{column_name}: {missing} пропусків "
                f"({percent:.2f}%)"
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()