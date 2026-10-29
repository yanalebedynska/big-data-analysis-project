from pyspark.sql.types import StructType, StructField, StringType


def get_crew_schema():
    return StructType([
        StructField("tconst", StringType(), True),
        StructField("directors", StringType(), True),
        StructField("writers", StringType(), True)
    ])


def read_crew(spark, file_path):
    schema = get_crew_schema()

    return spark.read \
        .option("header", True) \
        .option("sep", "\t") \
        .option("nullValue", "\\N") \
        .schema(schema) \
        .csv(file_path)
