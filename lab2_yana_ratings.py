from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType


def get_ratings_schema():
    return StructType([
        StructField("tconst", StringType(), True),
        StructField("averageRating", DoubleType(), True),
        StructField("numVotes", IntegerType(), True)
    ])


def read_ratings(spark, file_path):
    schema = get_ratings_schema()

    return spark.read \
        .option("header", True) \
        .option("sep", "\t") \
        .option("nullValue", "\\N") \
        .schema(schema) \
        .csv(file_path)
