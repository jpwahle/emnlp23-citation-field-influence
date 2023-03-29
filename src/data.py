from pyspark.sql import SparkSession

from config import get_spark_conf
from schemas import (
    get_original_citations_schema,
    get_original_papers_schema,
    get_truncated_citations_schema,
    get_truncated_papers_schema,
)


def get_spark_session():
    """
    Returns a SparkSession for performing data analysis on citation diversity.

    The function creates a SparkConf object, configures it, and uses it to create a SparkSession with the specified application name. If a SparkSession already exists, it returns the existing instance.

    Args:
        None

    Returns:
        pyspark.sql.SparkSession: The SparkSession object."""
    conf = get_spark_conf()
    return (
        SparkSession.builder.appName("Citation Diversity Analysis")
        .config(conf=conf)
        .getOrCreate()
    )


def load_papers_data_original(spark):
    """
    Loads the original papers data into a DataFrame using the specified SparkSession.

    The function reads the JSON files matching the pattern "papers.jsonl/*.jsonl" and applies the schema obtained from get_original_papers_schema().

    Args:
        spark (pyspark.sql.SparkSession): The SparkSession object.

    Returns:
        pyspark.sql.DataFrame: The DataFrame containing the original papers data.
    """

    return spark.read.json(
        "papers.jsonl/*.jsonl", schema=get_original_papers_schema()
    )


def load_citations_data_original(spark):
    """
    Loads the original citations data into a DataFrame using the specified SparkSession.

    The function reads the JSON files matching the pattern "citations/*.jsonl" and applies the schema obtained from get_original_citations_schema().

    Args:
        spark (pyspark.sql.SparkSession): The SparkSession object.

    Returns:
        pyspark.sql.DataFrame: The DataFrame containing the original citations data.
    """

    return spark.read.json(
        "citations/*.jsonl", schema=get_original_citations_schema()
    )


def load_papers_data(spark):
    """
    Loads the papers data into a DataFrame using the specified SparkSession.

    The function reads the JSON file "papers.jsonl" and applies the schema obtained from get_truncated_papers_schema().

    Args:
        spark (pyspark.sql.SparkSession): The SparkSession object.

    Returns:
        pyspark.sql.DataFrame: The DataFrame containing the papers data."""

    return spark.read.json(
        "papers.jsonl", schema=get_truncated_papers_schema()
    )


def load_citations_data(spark):
    """
    Loads the citations data into a DataFrame using the specified SparkSession.

    The function reads the JSON file "citations.jsonl" and applies the schema obtained from get_truncated_citations_schema().

    Args:
        spark (pyspark.sql.SparkSession): The SparkSession object.

    Returns:
        pyspark.sql.DataFrame: The DataFrame containing the citations data."""

    return spark.read.json(
        "citations.jsonl", schema=get_truncated_citations_schema()
    )


def write_papers_data(df):
    """
    Writes the papers data DataFrame to a JSON file.

    The function writes the DataFrame to the file "papers.jsonl" in overwrite mode.

    Args:
        df (pyspark.sql.DataFrame): The DataFrame containing the papers data.

    Returns:
        None"""

    df.write.json("papers.jsonl", mode="overwrite")


def write_citations_data(df):
    """
    Writes the citations data DataFrame to a JSON file.

    The function writes the DataFrame to the file "citations.jsonl" in overwrite mode.

    Args:
        df (pyspark.sql.DataFrame): The DataFrame containing the citations data.

    Returns:
        None"""

    df.write.json("citations.jsonl", mode="overwrite")
