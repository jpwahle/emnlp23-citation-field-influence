from pyspark import SparkConf


def get_spark_conf():
    """
    Returns a SparkConf object with the specified configuration settings.

    The function creates a SparkConf object and sets the following configuration properties:
    - spark.driver.port: "7070"
    - spark.driver.bindAddress: "0.0.0.0"
    - spark.ui.port: "4040"
    - spark.ui.reverseProxy: "true"

    Returns:
        pyspark.SparkConf: The SparkConf object."""

    conf = SparkConf()
    conf.set("spark.driver.port", "7070")
    conf.set("spark.driver.bindAddress", "0.0.0.0")
    conf.set("spark.ui.port", "4040")
    conf.set("spark.ui.reverseProxy", "true")
    return conf
