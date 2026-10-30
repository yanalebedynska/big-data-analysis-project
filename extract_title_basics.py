from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
)


#порядок полів відповідає стовпцям файлу title.basics.
TITLE_BASICS_SCHEMA = StructType([
    StructField("tconst", StringType(), True),
    StructField("titleType", StringType(), True),
    StructField("primaryTitle", StringType(), True),
    StructField("originalTitle", StringType(), True),
    StructField("isAdult", IntegerType(), True),
    StructField("startYear", IntegerType(), True),
    StructField("endYear", IntegerType(), True),
    StructField("runtimeMinutes", IntegerType(), True),
    StructField("genres", StringType(), True),
])


def extract_title_basics(
    spark: SparkSession,
    file_path: str,
) -> DataFrame:

    return (
        spark.read
        .schema(TITLE_BASICS_SCHEMA)
        .option("header", True)
        .option("sep", "\t")
        .option("nullValue", "\\N")
        .option("quote", "")
        .option("enforceSchema", False)
        .option("mode", "FAILFAST")
        .csv(file_path)
    )