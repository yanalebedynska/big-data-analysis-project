import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import date

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

        print("\n=== ПЕРЕВІРКА ДУБЛІКАТІВ ===")

        # Порожні ідентифікатори вже враховані серед пропусків.
        id_groups = (
            df.filter(F.col("tconst").isNotNull())
            .groupBy("tconst")
            .count()
        )

        id_statistics = id_groups.agg(
            F.count("*").alias("unique_ids"),
            F.count(
                F.when(F.col("count") > 1, 1)
            ).alias("repeated_ids"),
            F.coalesce(
                F.sum(F.col("count") - 1),
                F.lit(0),
            ).alias("extra_rows"),
        ).first()

        unique_rows = df.dropDuplicates().count()

        print(f"Загальна кількість рядків: {total_rows}")
        print(
            f"Унікальних непорожніх tconst: "
            f"{id_statistics['unique_ids']}"
        )
        print(
            f"Ідентифікаторів, які повторюються: "
            f"{id_statistics['repeated_ids']}"
        )
        print(
            f"Повторних рядків за tconst понад перший: "
            f"{id_statistics['extra_rows']}"
        )
        print(f"Повністю унікальних рядків: {unique_rows}")
        print(
            f"Повністю однакових рядків понад перший: "
            f"{total_rows - unique_rows}"
        )

        print("\n=== ПЕРЕВІРКА ЧИСЛОВИХ ЗНАЧЕНЬ ===")

        current_year = date.today().year
        long_runtime_limit = 24 * 60  # 1440 хвилин.

        checks = {
            "isAdult_not_0_or_1": ~F.col("isAdult").isin(0, 1),
            "startYear_nonpositive": F.col("startYear") <= 0,
            "endYear_nonpositive": F.col("endYear") <= 0,
            "endYear_before_startYear": (
                F.col("endYear") < F.col("startYear")
            ),
            "runtime_nonpositive": F.col("runtimeMinutes") <= 0,
            "startYear_in_future": (
                F.col("startYear") > current_year
            ),
            "endYear_in_future": (
                F.col("endYear") > current_year
            ),
            "runtime_over_24_hours": (
                F.col("runtimeMinutes") > long_runtime_limit
            ),
        }

        check_counts = df.agg(*[
            F.count(F.when(condition, 1)).alias(name)
            for name, condition in checks.items()
        ]).first()

        print(f"Поточний рік: {current_year}")
        print(f"Межа великої тривалості: {long_runtime_limit} хв")

        for name in checks:
            print(f"{name}: {check_counts[name]}")

        suspicious_condition = F.lit(False)

        for condition in checks.values():
            suspicious_condition = (
                suspicious_condition
                | F.coalesce(condition, F.lit(False))
            )

        print("\n=== ПРИКЛАДИ ЗАПИСІВ ДЛЯ ПЕРЕВІРКИ ===")

        (
            df.filter(suspicious_condition)
            .select(
                "tconst",
                "titleType",
                "primaryTitle",
                "isAdult",
                "startYear",
                "endYear",
                "runtimeMinutes",
            )
            .show(20, truncate=35)
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()