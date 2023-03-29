import os
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from pyspark.ml.feature import NGram, StopWordsRemover, Tokenizer
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    array_contains,
    array_join,
    avg,
    coalesce,
    col,
    collect_list,
    count,
    countDistinct,
    explode,
    expr,
    first,
    lit,
    lower,
    pandas_udf,
    percentile_approx,
    regexp_extract,
    regexp_replace,
    size,
    split,
    udf,
)
from pyspark.sql.types import ArrayType, FloatType, StringType

from .data import load_citations_data, load_papers_data


def compute_cs_subfield_to_nlp_subfield_citation_counts(
    citations_df, nlp_papers_df_with_categories, cs_papers_df
):
    """
    Computes the citation counts between Computer Science (CS) subfields and NLP subfields.

    The function performs the following steps:
    1. Calculates the citations from NLP papers to CS fields by joining the citations DataFrame with the NLP papers DataFrame with categories and the CS papers DataFrame.
    2. Groups the data by year, CS field, and NLP subfield, and counts the number of citations and distinct citing papers.
    3. Calculates the citations from CS fields to NLP papers by joining the citations DataFrame with the NLP papers DataFrame with categories and the CS papers DataFrame.
    4. Groups the data by year, CS field, and NLP subfield, and counts the number of citations and distinct cited papers.
    5. Combines the citation counts and paper counts for both directions of citation.
    6. Fills in missing values with zeros and renames the columns.
    7. Orders the result by year, CS field, and NLP subfield in descending order.
    8. Converts the result to a Pandas DataFrame.
    9. Saves the result to a CSV file.

    Args:
        citations_df (pyspark.sql.DataFrame): The DataFrame containing the citations data.
        nlp_papers_df_with_categories (pyspark.sql.DataFrame): The DataFrame containing the NLP papers data with categories.
        cs_papers_df (pyspark.sql.DataFrame): The DataFrame containing the CS papers data.

    Returns:
        None

    Example:
        ```python
        citations = spark.read.csv("citations.csv", header=True)
        nlp_papers = spark.read.csv("nlp_papers.csv", header=True)
        cs_papers = spark.read.csv("cs_papers.csv", header=True)
        compute_cs_subfield_to_nlp_subfield_citation_counts(citations, nlp_papers, cs_papers)
        ```
    """
    # Calculate citations from NLP papers to CS fields
    nlp_to_cs_citations = (
        citations_df.join(
            nlp_papers_df_with_categories,
            citations_df.citingcorpusid
            == nlp_papers_df_with_categories.corpusid,
        )
        .join(
            cs_papers_df, citations_df.citedcorpusid == cs_papers_df.corpusid
        )
        .select(
            nlp_papers_df_with_categories["year"],
            nlp_papers_df_with_categories["arr_categories"],
            cs_papers_df["cs_topics"],
            "citedcorpusid",
            "citingcorpusid",
        )
    )

    nlp_to_cs_citations = nlp_to_cs_citations.select(
        "year",
        explode("arr_categories").alias("arr_category"),
        "cs_topics",
        "citedcorpusid",
        "citingcorpusid",
    ).select(
        "year",
        "arr_category",
        explode("cs_topics").alias("cs_field"),
        "citedcorpusid",
        "citingcorpusid",
    )

    citations2to1 = nlp_to_cs_citations.groupby(
        "year", "cs_field", "arr_category"
    ).agg(
        count("citedcorpusid").alias("#citations1->2"),
        countDistinct("citingcorpusid").alias(
            "#papersfield1"
        ),  # Count distinct citing papers
    )

    # Calculate citations from CS fields to NLP papers
    cs_to_nlp_citations = (
        citations_df.join(
            nlp_papers_df_with_categories,
            citations_df.citedcorpusid
            == nlp_papers_df_with_categories.corpusid,
        )
        .join(
            cs_papers_df, citations_df.citingcorpusid == cs_papers_df.corpusid
        )
        .select(
            nlp_papers_df_with_categories["year"],
            nlp_papers_df_with_categories["arr_categories"],
            cs_papers_df["cs_topics"],
            "citedcorpusid",
            "citingcorpusid",
        )
    )

    cs_to_nlp_citations = cs_to_nlp_citations.select(
        "year",
        explode("arr_categories").alias("arr_category"),
        "cs_topics",
        "citedcorpusid",
        "citingcorpusid",
    ).select(
        "year",
        "arr_category",
        explode("cs_topics").alias("cs_field"),
        "citedcorpusid",
        "citingcorpusid",
    )

    citations1to2 = cs_to_nlp_citations.groupby(
        "year", "cs_field", "arr_category"
    ).agg(
        count("citedcorpusid").alias("#citations2->1"),
        countDistinct("citingcorpusid").alias(
            "#papersfield2"
        ),  # Count distinct cited papers
    )

    # Combine citation counts, paper counts
    final_result = (
        citations1to2.join(
            citations2to1,
            on=["year", "cs_field", "arr_category"],
            how="full_outer",
        )
        .withColumn("#citations1->2", coalesce(col("#citations1->2"), lit(0)))
        .withColumn("#citations2->1", coalesce(col("#citations2->1"), lit(0)))
        .withColumn("#papersfield1", coalesce(col("#papersfield1"), lit(0)))
        .withColumn("#papersfield2", coalesce(col("#papersfield2"), lit(0)))
        .select(
            col("year"),
            col("cs_field").alias("field1"),
            col("arr_category").alias("field2"),
            col("#citations1->2"),
            col("#citations2->1"),
            col("#papersfield1"),
            col("#papersfield2"),
        )
    )

    # Order the result
    final_result = final_result.orderBy("year", "field1", ascending=False)

    # Convert to pandas
    final_result_pd = final_result.toPandas()

    # Save the result to a CSV file
    final_result_pd.to_csv("./outputs/citations_cs_fields_to_arr_by_year.csv")


def compute_cs_subfield_to_nlp_citation_counts(
    citations_df, nlp_papers_df, cs_papers_df
):
    """
    Computes the citation counts between Computer Science (CS) subfields and NLP papers.

    The function performs the following steps:
    1. Calculates the citations from NLP papers to CS fields by joining the citations DataFrame with the NLP papers DataFrame and the CS papers DataFrame.
    2. Groups the data by year and CS field, and counts the number of citations and distinct citing papers.
    3. Calculates the citations from CS fields to NLP papers by joining the citations DataFrame with the NLP papers DataFrame and the CS papers DataFrame.
    4. Groups the data by year and CS field, and counts the number of citations and distinct cited papers.
    5. Combines the citation counts and paper counts for both directions of citation.
    6. Fills in missing values with zeros and renames the columns.
    7. Orders the result by year and field in descending order.
    8. Converts the result to a Pandas DataFrame.
    9. Saves the result to a CSV file.

    Args:
        citations_df (pyspark.sql.DataFrame): The DataFrame containing the citations data.
        nlp_papers_df (pyspark.sql.DataFrame): The DataFrame containing the NLP papers data.
        cs_papers_df (pyspark.sql.DataFrame): The DataFrame containing the CS papers data.

    Returns:
        None

    Example:
        ```python
        citations = spark.read.csv("citations.csv", header=True)
        nlp_papers = spark.read.csv("nlp_papers.csv", header=True)
        cs_papers = spark.read.csv("cs_papers.csv", header=True)
        compute_cs_subfield_to_nlp_citation_counts(citations, nlp_papers, cs_papers)
        ```
    """

    # Calculate citations from NLP papers to CS fields
    nlp_to_cs_citations = (
        citations_df.join(
            nlp_papers_df,
            citations_df.citingcorpusid == nlp_papers_df.corpusid,
        )
        .join(
            cs_papers_df, citations_df.citedcorpusid == cs_papers_df.corpusid
        )
        .select(
            nlp_papers_df["year"],
            cs_papers_df["cs_topics"],
            "citedcorpusid",
            "citingcorpusid",
        )
    )

    nlp_to_cs_citations = nlp_to_cs_citations.select(
        "year",
        explode("cs_topics").alias("cs_field"),
        "citedcorpusid",
        "citingcorpusid",
    )

    citations2to1 = nlp_to_cs_citations.groupby("year", "cs_field").agg(
        count("citedcorpusid").alias("#citations1->2"),
        countDistinct("citingcorpusid").alias(
            "#papersfield1"
        ),  # Count distinct citing papers
    )

    # Calculate citations from CS fields to NLP papers
    cs_to_nlp_citations = (
        citations_df.join(
            nlp_papers_df, citations_df.citedcorpusid == nlp_papers_df.corpusid
        )
        .join(
            cs_papers_df, citations_df.citingcorpusid == cs_papers_df.corpusid
        )
        .select(
            nlp_papers_df["year"],
            cs_papers_df["cs_topics"],
            "citedcorpusid",
            "citingcorpusid",
        )
    )

    cs_to_nlp_citations = cs_to_nlp_citations.select(
        "year",
        explode("cs_topics").alias("cs_field"),
        "citedcorpusid",
        "citingcorpusid",
    )

    citations1to2 = cs_to_nlp_citations.groupby("year", "cs_field").agg(
        count("citedcorpusid").alias("#citations2->1"),
        countDistinct("citingcorpusid").alias(
            "#papersfield2"
        ),  # Count distinct cited papers
    )

    # Combine citation counts, paper counts
    final_result = (
        citations1to2.join(
            citations2to1, on=["year", "cs_field"], how="full_outer"
        )
        .withColumn("#citations1->2", coalesce(col("#citations1->2"), lit(0)))
        .withColumn("#citations2->1", coalesce(col("#citations2->1"), lit(0)))
        .withColumn("#papersfield1", coalesce(col("#papersfield1"), lit(0)))
        .withColumn("#papersfield2", coalesce(col("#papersfield2"), lit(0)))
        .select(
            col("year"),
            col("cs_field").alias("field1"),
            lit("NLP").alias("field2"),
            col("#citations1->2"),
            col("#citations2->1"),
            col("#papersfield1"),
            col("#papersfield2"),
        )
    )

    # Order the result
    final_result = final_result.orderBy("year", "field1", ascending=False)

    # Convert to pandas
    final_result_pd = final_result.toPandas()

    # Save the result to a CSV file
    final_result_pd.to_csv("./outputs/citations_cs_fields_to_nlp_by_year.csv")


def compute_cs_subfields(papers_df):
    """
    Computes the subfields of Computer Science (CS) for the given DataFrame of papers.

    The function performs the following steps:
    1. Reads the enhanced papers from a JSON file and joins them with the input papers DataFrame based on the corpusid.
    2. Loads the ontology CSV file and applies URL decoding to the columns.
    3. Extracts the relevant parts of the ontology URLs and replaces underscores with spaces.
    4. Filters the ontology DataFrame to include only rows where the second column contains 'superTopicOf'.
    5. Converts the ontology DataFrame to a dictionary, adding notable entries from AI and removing old entries.
    6. Removes categories and their siblings based on a predefined list of categories to remove.
    7. Broadcasts the ontology dictionary for efficient lookup.
    8. Defines a UDF to find the first level topics from a given list of topics based on a maximum depth.
    9. Applies the UDF to find the first-level topics for each paper in the DataFrame.
    10. Flattens the s2fieldsofstudy array and filters the rows where the category is "Computer Science".

    Args:
        papers_df (pyspark.sql.DataFrame): The DataFrame containing the papers data.

    Returns:
        pyspark.sql.DataFrame: The DataFrame of papers with the subfields of Computer Science.

    Example:
        ```python
        papers = spark.read.csv("papers.csv", header=True)
        cs_papers = compute_cs_subfields(papers)
        cs_papers.show()
        ```"""

    # Define the schema for the enhanced papers
    enhanced_papers_schema = StructType(
        [
            StructField("corpusid", IntegerType(), True),
            StructField("enhanced", ArrayType(StringType()), True),
        ]
    )

    # Read the enhanced papers
    enhanced_papers_df = spark.read.json(
        "outputs/2022-11-30-papers.jsonl", schema=enhanced_papers_schema
    )

    # Join the papers_df with enhanced_papers_df on corpusid
    papers_df = papers_df.join(enhanced_papers_df, on="corpusid", how="left")

    # Load the ontology CSV
    ontology_df = spark.read.csv(
        "outputs/CSO.3.3.csv", inferSchema=True, header=False
    )

    # Define UDF to URL decode strings
    url_decode = udf(lambda url: urllib.parse.unquote(url), StringType())

    # Remove '<' and '>' symbols
    ontology_df = ontology_df.withColumn(
        "_c0", regexp_replace("_c0", "[<>]", "")
    )
    ontology_df = ontology_df.withColumn(
        "_c1", regexp_replace("_c1", "[<>]", "")
    )
    ontology_df = ontology_df.withColumn(
        "_c2", regexp_replace("_c2", "[<>]", "")
    )

    # Now decode URL
    ontology_df = ontology_df.withColumn("_c0", url_decode(ontology_df._c0))
    ontology_df = ontology_df.withColumn("_c1", url_decode(ontology_df._c1))
    ontology_df = ontology_df.withColumn("_c2", url_decode(ontology_df._c2))

    # Now proceed with the previous operations
    ontology_df = ontology_df.withColumn(
        "_c0", split(ontology_df._c0, "/").getItem(4)
    )
    ontology_df = ontology_df.withColumn(
        "_c2", split(ontology_df._c2, "/").getItem(4)
    )
    ontology_df = ontology_df.withColumn(
        "_c1", split(ontology_df._c1, "/").getItem(4)
    )

    # Replace "_" with " "
    ontology_df = ontology_df.withColumn(
        "_c0", regexp_replace("_c0", "_", " ")
    )
    ontology_df = ontology_df.withColumn(
        "_c2", regexp_replace("_c2", "_", " ")
    )

    # Filter rows where '_c1' contains 'superTopicOf'
    ontology_df = ontology_df.filter((col("_c1").contains("superTopicOf")))

    # Convert the DataFrame to a dictionary
    ontology_dict = {row["_c2"]: row["_c0"] for row in ontology_df.collect()}

    # Add notable entries from AI
    ontology_dict["machine learning"] = "computer science"
    ontology_dict["natural language processing"] = "computer science"
    ontology_dict["expert systems"] = "computer science"
    ontology_dict["genetic algorithms"] = "computer science"

    # Remove old entries if they exist
    if (
        "machine learning" in ontology_dict
        and ontology_dict["machine learning"] == "artificial intelligence"
    ):
        del ontology_dict["machine learning"]
    if (
        "natural language processing" in ontology_dict
        and ontology_dict["natural language processing"]
        == "artificial intelligence"
    ):
        del ontology_dict["natural language processing"]
    if (
        "expert systems" in ontology_dict
        and ontology_dict["expert systems"] == "artificial intelligence"
    ):
        del ontology_dict["expert systems"]
    if (
        "genetic algorithms" in ontology_dict
        and ontology_dict["genetic algorithms"] == "artificial intelligence"
    ):
        del ontology_dict["genetic algorithms"]

    # Categories to remove because of low counts (see CSO 15 highly published categories)
    categories_to_remove = [
        "computer-aided design",
        "computer hardware",
        "computer security",
        "human computer interaction",
        "theoretical computer science",
    ]

    # Remove categories and their siblings
    for category in categories_to_remove:
        siblings = []
        for key, value in ontology_dict.items():
            if value == category:
                siblings.append(key)
        for sibling in siblings:
            del ontology_dict[sibling]
        if category in ontology_dict:
            del ontology_dict[category]

    # Broadcast the dictionary
    ontology_bc = spark.sparkContext.broadcast(ontology_dict)

    def find_first_level_topics(topics, max_depth=100):
        """
        Finds the first level topics from a given list of topics based on a maximum depth.

        The function iterates over each topic and checks its super topics up to the maximum depth. If a super topic is found that matches "computer science", the topic is considered a first level topic and added to the result list.

        Args:
            topics (list): The list of topics to search for first level topics.
            max_depth (int, optional): The maximum depth to search for super topics. Defaults to 100.

        Returns:
            list: The list of first level topics found.

        Example:
            ```python
            topics = ["machine_learning", "deep_learning", "natural_language_processing", "algorithms"]
            first_level_topics = find_first_level_topics(topics, max_depth=10)
            print(first_level_topics)
            # Output: ["machine_learning", "deep_learning"]
            ```"""

        first_level_topics = []
        if topics is None:
            return first_level_topics
        for topic in topics:
            depth = 0
            while depth < max_depth:
                super_topic = ontology_bc.value.get(topic.replace("_", " "))
                if super_topic is None:
                    break
                elif super_topic == "computer science":
                    first_level_topics.append(topic)
                    break
                topic = super_topic
                depth += 1
        return list(
            set(first_level_topics)
        )  # Remove duplicates and return as a list

    # Convert the function to a UDF
    find_first_level_topics_udf = udf(
        find_first_level_topics, ArrayType(StringType())
    )

    # Apply the UDF to find first-level topics
    papers_df = papers_df.withColumn(
        "cs_topics", find_first_level_topics_udf("enhanced")
    )

    # First, we need to flatten the s2fieldsofstudy array
    papers_df_flat = papers_df.withColumn(
        "s2fieldsofstudy", explode("s2fieldsofstudy")
    )

    # Then, we can filter the rows where category is "Computer Science"
    cs_papers_df = papers_df_flat.filter(
        col("s2fieldsofstudy.category") == "Computer Science"
    )

    return cs_papers_df


def compute_num_fields_per_paper(nlp_papers_df, citations_df):
    """
    Computes the average and median number of fields of study per paper that cites or is cited by NLP papers over time.

    The function performs the following steps:
    1. Joins the NLP papers DataFrame with the citations DataFrame to get the fields of study for citing and cited papers.
    2. Computes the average and median number of fields of study for citing papers.
    3. Computes the average and median number of fields of study for cited papers.
    4. Combines the statistics for citing and cited papers.
    5. Converts the result to a Pandas DataFrame and saves it to a CSV file.

    Args:
        nlp_papers_df (pyspark.sql.DataFrame): The DataFrame containing the NLP papers data.
        citations_df (pyspark.sql.DataFrame): The DataFrame containing the citations data.

    Returns:
        None"""

    # Join NLP papers with citations to get citing papers' fields of study
    citing_df = (
        nlp_papers_df.alias("nlp")
        .join(
            citations_df.alias("cit"),
            col("nlp.corpusid") == col("cit.citingcorpusid"),
            "inner",
        )
        .select("cit.citingcorpusid", "nlp.year", "nlp.s2fieldsofstudy")
    )

    # Join NLP papers with citations to get cited papers' fields of study
    cited_df = (
        nlp_papers_df.alias("nlp")
        .join(
            citations_df.alias("cit"),
            col("nlp.corpusid") == col("cit.citedcorpusid"),
            "inner",
        )
        .select("cit.citedcorpusid", "nlp.year", "nlp.s2fieldsofstudy")
    )

    # Compute average and median number of fields of study for citing papers
    citing_fields_stats = (
        citing_df.filter(
            size("s2fieldsofstudy") > 0
        )  # Filter out empty arrays
        .groupBy("year")
        .agg(
            avg(size("s2fieldsofstudy")).alias("avg_citing_fields"),
            percentile_approx(size("s2fieldsofstudy"), 0.5).alias(
                "median_citing_fields"
            ),
        )
        .orderBy("year")
    )

    # Compute average and median number of fields of study for cited papers
    cited_fields_stats = (
        cited_df.filter(size("s2fieldsofstudy") > 0)  # Filter out empty arrays
        .groupBy("year")
        .agg(
            avg(size("s2fieldsofstudy")).alias("avg_cited_fields"),
            percentile_approx(size("s2fieldsofstudy"), 0.5).alias(
                "median_cited_fields"
            ),
        )
        .orderBy("year")
    )

    # Combine citing and cited fields statistics
    fields_stats_df = citing_fields_stats.join(
        cited_fields_stats, on="year", how="outer"
    ).orderBy("year")

    fields_stats_pd = fields_stats_df.toPandas()
    fields_stats_pd.to_csv(
        "./outputs/mean_median_fields_a_paper_that_cites_cited_NLP_has_over_time.csv"
    )


def compute_cfdi_nlp_by_citation_quantile(nlp_papers_df, papers_df):
    """
    Computes the CFDI (Citational Field Diversity Index) for NLP papers by citation quantile.

    The function performs the following steps:
    1. Defines a UDF (User-Defined Function) to calculate the CFDI for a given series.
    2. Filters the NLP papers DataFrame and the papers DataFrame to include papers with a year greater than 1965.
    3. Computes the citation quantiles for the NLP papers.
    4. Adds a quantile column to the NLP papers DataFrame based on the citation quantiles.
    5. Calculates citations from other categories to NLP papers.
    6. Calculates citations from NLP papers to other categories.
    7. Groups the incoming and outgoing citations by corpusid, year, title, authors, citation quantile, citation count, and category, and counts the occurrences.
    8. Applies the CFDI UDF to calculate the diversity scores for incoming and outgoing citations.
    9. Joins the incoming and outgoing diversity scores.
    10. Converts the result to a Pandas DataFrame and saves it to a CSV file.

    Args:
        nlp_papers_df (pyspark.sql.DataFrame): The DataFrame containing the NLP papers data.
        papers_df (pyspark.sql.DataFrame): The DataFrame containing the papers data.

    Returns:
        None"""

    def cfdi(series):
        """
        Calculates the CFDI (Citational Field Diversity Index) for a given series.

        The function calculates the diversity index using the formula: 1 - sum((values / n) ** 2), where n is the sum of values in the series.

        Args:
            series (pandas.Series): The input series for which to calculate the diversity index.

        Returns:
            pandas.Series: The series with the calculated diversity index for each element.
        """

        def calculate_diversity(values):
            """
            Calculates the diversity index for a given array of values.

            The function calculates the diversity index using the formula: 1 - sum((values / n) ** 2), where n is the sum of values in the array.

            Args:
                values (array-like): The input array of values for which to calculate the diversity index.

            Returns:
                float: The calculated diversity index.
            """
            n = np.sum(values)
            return 1 - np.sum((values / n) ** 2)

        return series.apply(calculate_diversity)

    cfdi_udf = pandas_udf(cfdi, FloatType())

    # Filter papers with year > 1965
    nlp_papers = nlp_papers_df.filter("year > 1965")
    papers_df = papers_df.filter("year > 1965")

    # Compute percentiles
    percentiles = nlp_papers.approxQuantile(
        "citationcount", [i / 10 for i in range(1, 10)], 0.001
    )

    # Add quantile column
    nlp_papers = nlp_papers.withColumn(
        "citationquantile",
        expr(
            "case when citationcount <= {} then 'q10' when citationcount <= {}"
            " then 'q20' when citationcount <= {} then 'q30' when"
            " citationcount <= {} then 'q40' when citationcount <= {} then"
            " 'q50' when citationcount <= {} then 'q60' when citationcount <="
            " {} then 'q70' when citationcount <= {} then 'q80' when"
            " citationcount <= {} then 'q90' else 'q100' end".format(
                *percentiles
            )
        ),
    )

    # Calculate citations from other categories to NLP papers
    nlp_in_citations = (
        citations_df.join(
            nlp_papers,
            citations_df.citedcorpusid == nlp_papers.corpusid,
            how="inner",
        )
        .join(
            papers_df.alias("p2"),
            citations_df.citingcorpusid == F.col("p2.corpusid"),
            how="inner",
        )
        .select(
            nlp_papers["corpusid"],
            nlp_papers["year"],
            nlp_papers["title"],
            nlp_papers["authors"],
            nlp_papers["citationcount"],
            "citationquantile",
            F.col("p2.s2fieldsofstudy").alias("s2fieldsofstudy"),
        )
    )

    # Calculate citations from NLP papers to other categories
    nlp_out_citations = (
        citations_df.join(
            nlp_papers,
            citations_df.citingcorpusid == nlp_papers.corpusid,
            how="inner",
        )
        .join(
            papers_df.alias("p2"),
            citations_df.citedcorpusid == F.col("p2.corpusid"),
            how="inner",
        )
        .select(
            nlp_papers["corpusid"],
            nlp_papers["year"],
            nlp_papers["title"],
            nlp_papers["authors"],
            nlp_papers["citationcount"],
            "citationquantile",
            F.col("p2.s2fieldsofstudy").alias("s2fieldsofstudy"),
        )
    )

    # Explode the 's2fieldsofstudy' column to get individual categories
    grouped_incoming = (
        nlp_in_citations.select(
            "corpusid",
            "year",
            "title",
            "authors",
            "citationquantile",
            "citationcount",
            F.explode("s2fieldsofstudy").alias("category"),
        )
        .groupBy(
            "corpusid",
            "year",
            "title",
            "authors",
            "citationquantile",
            "citationcount",
            "category",
        )
        .count()
    )
    grouped_outgoing = (
        nlp_out_citations.select(
            "corpusid",
            "year",
            "title",
            "authors",
            "citationquantile",
            "citationcount",
            F.explode("s2fieldsofstudy").alias("category"),
        )
        .groupBy(
            "corpusid",
            "year",
            "title",
            "authors",
            "citationquantile",
            "citationcount",
            "category",
        )
        .count()
    )

    # Apply CFDI function
    incoming_diversity = grouped_incoming.groupby(
        "corpusid",
        "year",
        "title",
        "authors",
        "citationquantile",
        "citationcount",
    ).agg(cfdi_udf(F.collect_list("count")).alias("incoming_diversity"))
    outgoing_diversity = grouped_outgoing.groupby(
        "corpusid",
        "year",
        "title",
        "authors",
        "citationquantile",
        "citationcount",
    ).agg(cfdi_udf(F.collect_list("count")).alias("outgoing_diversity"))

    # Join the incoming and outgoing diversity scores
    diversity_df = incoming_diversity.join(
        outgoing_diversity,
        on=[
            "corpusid",
            "year",
            "title",
            "authors",
            "citationquantile",
            "citationcount",
        ],
        how="full_outer",
    )

    # Convert to Pandas DataFrame
    diversity_pd = diversity_df.toPandas()
    diversity_pd.to_csv("outputs/nlp_papers_diversity.csv")


def compute_cfdi_non_cs_fields(papers_df, citations_df):
    """
    Computes the CFDI (Citational Field Diversity Index) for non-CS fields by year.

    The function performs the following steps:
    1. Defines a UDF (User-Defined Function) to calculate the CFDI for a given series.
    2. Extracts all unique categories from the papers DataFrame.
    3. Creates an empty DataFrame to store the results.
    4. Iterates over each category and filters the papers DataFrame for the current category.
    5. Calculates citations from other categories to papers in the current category.
    6. Calculates citations from papers in the current category to other categories.
    7. Groups the incoming and outgoing citations by corpusid, year, and category, and counts the occurrences.
    8. Applies the CFDI UDF to calculate the diversity scores for incoming and outgoing citations.
    9. Joins the incoming and outgoing diversity scores.
    10. Calculates the average diversity per year for the current category.
    11. Converts the result to a Pandas DataFrame and appends it to the final_diversity_all DataFrame.
    12. Saves the final results to a CSV file.

    Args:
        papers_df (pyspark.sql.DataFrame): The DataFrame containing the papers data.
        citations_df (pyspark.sql.DataFrame): The DataFrame containing the citations data.

    Returns:
        None"""

    def cfdi(series):
        """
        Calculates the CFDI (Citational Field Diversity Index) for a given series.

        The function calculates the diversity index using the formula: 1 - sum((values / n) ** 2), where n is the sum of values in the series.

        Args:
            series (pandas.Series): The input series for which to calculate the diversity index.

        Returns:
            pandas.Series: The series with the calculated diversity index for each element.
        """

        def calculate_diversity(values):
            """
            Calculates the diversity index for a given array of values.

            The function calculates the diversity index using the formula: 1 - sum((values / n) ** 2), where n is the sum of values in the array.

            Args:
                values (array-like): The input array of values for which to calculate the diversity index.

            Returns:
                float: The calculated diversity index.
            """
            n = np.sum(values)
            return 1 - np.sum((values / n) ** 2)

        return series.apply(calculate_diversity)

    cfdi_udf = pandas_udf(cfdi, FloatType())

    # Extract all unique categories
    all_categories = (
        papers_df.select(
            F.explode("s2fieldsofstudy.category").alias("category")
        )
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    # Create an empty DataFrame to store results
    final_diversity_all = pd.DataFrame()

    for category in all_categories:
        # Filter papers for the current category
        category_papers = papers_df.filter(
            F.array_contains(papers_df.s2fieldsofstudy.category, category)
        ).alias("category_papers")

        # Calculate citations from other categories to category papers
        category_in_citations = (
            citations_df.join(
                category_papers,
                citations_df.citedcorpusid == category_papers.corpusid,
                how="inner",
            )
            .join(
                papers_df.alias("p2"),
                citations_df.citingcorpusid == F.col("p2.corpusid"),
                how="inner",
            )
            .select(
                category_papers["corpusid"],
                category_papers["year"],
                F.col("p2.s2fieldsofstudy").alias("s2fieldsofstudy"),
            )
        )

        # Calculate citations from category papers to other categories
        category_out_citations = (
            citations_df.join(
                category_papers,
                citations_df.citingcorpusid == category_papers.corpusid,
                how="inner",
            )
            .join(
                papers_df.alias("p2"),
                citations_df.citedcorpusid == F.col("p2.corpusid"),
                how="inner",
            )
            .select(
                category_papers["corpusid"],
                category_papers["year"],
                F.col("p2.s2fieldsofstudy").alias("s2fieldsofstudy"),
            )
        )

        # Explode the 's2fieldsofstudy' column to get individual categories
        grouped_incoming = (
            category_in_citations.select(
                "corpusid",
                "year",
                F.explode("s2fieldsofstudy").alias("category"),
            )
            .groupBy("corpusid", "year", "category")
            .count()
        )
        grouped_outgoing = (
            category_out_citations.select(
                "corpusid",
                "year",
                F.explode("s2fieldsofstudy").alias("category"),
            )
            .groupBy("corpusid", "year", "category")
            .count()
        )

        # Apply CFDI function
        incoming_diversity = grouped_incoming.groupby("corpusid", "year").agg(
            cfdi_udf(F.collect_list("count")).alias("incoming_diversity")
        )
        outgoing_diversity = grouped_outgoing.groupby("corpusid", "year").agg(
            cfdi_udf(F.collect_list("count")).alias("outgoing_diversity")
        )

        # Join the incoming and outgoing diversity scores
        diversity_category = incoming_diversity.join(
            outgoing_diversity, on=["corpusid", "year"], how="full_outer"
        )

        # Average CFDI per year
        avg_diversity_category = (
            diversity_category.groupby("year")
            .agg(
                F.avg("incoming_diversity").alias("avg_incoming_diversity"),
                F.avg("outgoing_diversity").alias("avg_outgoing_diversity"),
            )
            .select(
                F.col("year"),
                F.lit(category).alias("field"),
                F.col("avg_incoming_diversity"),
                F.col("avg_outgoing_diversity"),
            )
        )

        # Convert to pandas and append to final_diversity_all
        avg_diversity_category_pd = avg_diversity_category.orderBy(
            "year", ascending=False
        ).toPandas()
        final_diversity_all = pd.concat(
            [final_diversity_all, avg_diversity_category_pd], ignore_index=True
        )

    # Save the final results to a CSV file
    final_diversity_all.to_csv(
        "./outputs/avg_diversity_all_fields_by_year.csv"
    )


def compute_non_cs_fields_and_non_cs_fields_citation_counts(
    papers_df, citations_df
):
    """
    Computes citation counts from non-CS fields to non-CS fields by year.

    The function performs the following steps:
    1. Extracts all unique categories from the papers DataFrame.
    2. Creates an empty DataFrame to store the results.
    3. Iterates over each category and filters the papers DataFrame for the current category.
    4. Calculates citations from papers in the current category to other categories.
    5. Aggregates the citation counts and paper counts from papers in the current category to other categories.
    6. Calculates citations from other categories to papers in the current category.
    7. Aggregates the citation counts and paper counts from other categories to papers in the current category.
    8. Combines the citation counts, paper counts, and category paper counts.
    9. Converts the result to a Pandas DataFrame and appends it to the final_results_all DataFrame.
    10. Saves the final results to a CSV file.

    Args:
        papers_df (pyspark.sql.DataFrame): The DataFrame containing the papers data.
        citations_df (pyspark.sql.DataFrame): The DataFrame containing the citations data.

    Returns:
        None
    """

    # Extract all unique categories
    all_categories = (
        papers_df.select(explode("s2fieldsofstudy.category").alias("category"))
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    # Create an empty DataFrame to store results
    final_results_all = pd.DataFrame()

    for category in all_categories:
        # Filter papers for the current category
        category_papers = papers_df.filter(
            array_contains(papers_df.s2fieldsofstudy.category, category)
        ).alias("category_papers")

        # Calculate citations from category papers to other categories
        out_citations_category = (
            citations_df.join(
                category_papers,
                citations_df.citingcorpusid == category_papers.corpusid,
            )
            .join(
                papers_df.alias("papers_df"),
                citations_df.citedcorpusid == col("papers_df.corpusid"),
            )
            .select(
                category_papers["year"],
                explode(col("papers_df.s2fieldsofstudy")).alias(
                    "s2fieldsofstudy"
                ),
                "citedcorpusid",
                "citingcorpusid",
            )
            .select(
                "year",
                col("s2fieldsofstudy.category").alias("category"),
                "citedcorpusid",
                "citingcorpusid",
            )
        )

        citations2to1_category = out_citations_category.groupby(
            "year", "category"
        ).agg(
            count("citedcorpusid").alias("#citations1->2"),
            countDistinct("citingcorpusid").alias(
                "#papersfield1"
            ),  # Count distinct citing papers
        )

        # Calculate citations from other categories to category papers
        in_citations_category = (
            citations_df.join(
                category_papers,
                citations_df.citedcorpusid == category_papers.corpusid,
            )
            .join(
                papers_df.alias("papers_df"),
                citations_df.citingcorpusid == col("papers_df.corpusid"),
            )
            .select(
                category_papers["year"],
                explode(col("papers_df.s2fieldsofstudy")).alias(
                    "s2fieldsofstudy"
                ),
                "citedcorpusid",
                "citingcorpusid",
            )
            .select(
                "year",
                col("s2fieldsofstudy.category").alias("category"),
                "citedcorpusid",
                "citingcorpusid",
            )
        )

        citations1to2_category = in_citations_category.groupby(
            "year", "category"
        ).agg(
            count("citedcorpusid").alias("#citations2->1"),
            countDistinct("citingcorpusid").alias(
                "#papersfield2"
            ),  # Count distinct cited papers
        )

        # Combine citation counts, paper counts, and category paper counts
        result_category = (
            citations1to2_category.join(
                citations2to1_category,
                on=["year", "category"],
                how="full_outer",
            )
            .withColumn(
                "#citations1->2", coalesce(col("#citations1->2"), lit(0))
            )
            .withColumn(
                "#citations2->1", coalesce(col("#citations2->1"), lit(0))
            )
            .withColumn(
                "#papersfield1", coalesce(col("#papersfield1"), lit(0))
            )
            .withColumn(
                "#papersfield2", coalesce(col("#papersfield2"), lit(0))
            )
            .select(
                col("year"),
                col("category").alias("field1"),
                lit(category).alias("field2"),
                col("#citations1->2"),
                col("#citations2->1"),
                col("#papersfield1"),
                col("#papersfield2"),
            )
        )

        # Convert to pandas and append to final_results_all
        result_category_pd = result_category.orderBy(
            "year", "field1", ascending=False
        ).toPandas()
        # final_results_all = final_results_all.append(result_category_pd)
        final_results_all = pd.concat(
            [final_results_all, result_category_pd], ignore_index=True
        )

    # Save the final results to a CSV file
    final_results_all.to_csv(
        "./outputs/citations_all_non_cs_fields_by_year.csv"
    )


def compute_non_cs_fields_and_nlp_citation_counts(
    nlp_papers_df, papers_df, citations_df
):
    """
    Computes citation counts from non-CS fields to NLP by year.

    The function performs the following steps:
    1. Filters the nlp_papers_df and papers_df DataFrames to include papers with a year greater than 1965.
    2. Caches the nlp_papers DataFrame for optimization.
    3. Calculates citations from NLP papers to other categories.
    4. Aggregates the citation counts and paper counts from NLP papers to other categories.
    5. Calculates citations from other categories to NLP papers.
    6. Aggregates the citation counts and paper counts from other categories to NLP papers.
    7. Combines the citation counts, paper counts, and NLP paper counts.
    8. Orders the result by year and field categories.
    9. Converts the result to a Pandas DataFrame.
    10. Saves the result to a CSV file.

    Args:
        nlp_papers_df (pyspark.sql.DataFrame): The DataFrame containing the NLP papers data.
        papers_df (pyspark.sql.DataFrame): The DataFrame containing the papers data.
        citations_df (pyspark.sql.DataFrame): The DataFrame containing the citations data.

    Returns:
        None
    """
    # Filter papers with year > 1965
    nlp_papers = nlp_papers_df.filter("year > 1965")
    papers_df = papers_df.filter("year > 1965")

    # Cache the nlp_papers DataFrame
    nlp_papers = nlp_papers.cache()

    # Calculate citations from NLP papers to other categories
    nlp_out_citations = (
        citations_df.join(
            nlp_papers, citations_df.citingcorpusid == nlp_papers.corpusid
        )
        .join(papers_df, citations_df.citedcorpusid == papers_df.corpusid)
        .select(
            nlp_papers["year"],
            explode(papers_df["s2fieldsofstudy"]).alias("s2fieldsofstudy"),
            "citedcorpusid",
            "citingcorpusid",
        )
        .select(
            "year",
            col("s2fieldsofstudy.category").alias("category"),
            "citedcorpusid",
            "citingcorpusid",
        )
    )

    citations2to1 = nlp_out_citations.groupby("year", "category").agg(
        count("citedcorpusid").alias("#citations1->2"),
        countDistinct("citingcorpusid").alias(
            "#papersfield1"
        ),  # Count distinct citing papers
    )

    # Calculate citations from other categories to NLP papers
    nlp_in_citations = (
        citations_df.join(
            nlp_papers, citations_df.citedcorpusid == nlp_papers.corpusid
        )
        .join(papers_df, citations_df.citingcorpusid == papers_df.corpusid)
        .select(
            nlp_papers["year"],
            explode(papers_df["s2fieldsofstudy"]).alias("s2fieldsofstudy"),
            "citedcorpusid",
            "citingcorpusid",
        )
        .select(
            "year",
            col("s2fieldsofstudy.category").alias("category"),
            "citedcorpusid",
            "citingcorpusid",
        )
    )

    citations1to2 = nlp_in_citations.groupby("year", "category").agg(
        count("citedcorpusid").alias("#citations2->1"),
        countDistinct("citingcorpusid").alias(
            "#papersfield2"
        ),  # Count distinct cited papers
    )

    # Combine citation counts, paper counts, and NLP paper counts
    final_result = (
        citations1to2.join(
            citations2to1, on=["year", "category"], how="full_outer"
        )
        .withColumn("#citations1->2", coalesce(col("#citations1->2"), lit(0)))
        .withColumn("#citations2->1", coalesce(col("#citations2->1"), lit(0)))
        .withColumn("#papersfield1", coalesce(col("#papersfield1"), lit(0)))
        .withColumn("#papersfield2", coalesce(col("#papersfield2"), lit(0)))
        .select(
            col("year"),
            col("category").alias("field1"),
            lit("NLP").alias("field2"),
            col("#citations1->2"),
            col("#citations2->1"),
            col("#papersfield1"),
            col("#papersfield2"),
        )
    )

    # Order the result
    final_result = final_result.orderBy("year", "field1", ascending=False)

    # Convert to pandas
    final_result_pd = final_result.toPandas()

    # Save the result to a CSV file
    final_result_pd.to_csv(
        "./outputs/citations_non_cs_fields_to_nlp_by_year.csv"
    )


def compute_nlp_subfields_to_non_cs_fields_citation_counts(
    citations_df, nlp_papers_df_with_categories, papers_df
):
    """
    Computes citation counts from NLP subfields to non-CS fields by year.

    The function performs the following steps:
    1. Joins the citations DataFrame with the nlp_papers_df_with_categories DataFrame and papers_df to get the ARR papers' categories and the cited papers' fields of study.
    2. Filters the joined DataFrame to include only citations from ARR papers to other categories.
    3. Aggregates the citation counts and paper counts from ARR papers to other categories.
    4. Joins the citations DataFrame with the nlp_papers_df_with_categories DataFrame and papers_df to get the ARR papers' categories and the citing papers' fields of study.
    5. Filters the joined DataFrame to include only citations from other categories to ARR papers.
    6. Aggregates the citation counts and paper counts from other categories to ARR papers.
    7. Combines the citation counts, paper counts, and ARR paper counts.
    8. Orders the result by year and field categories.
    9. Converts the result to a Pandas DataFrame.
    10. Saves the result to a CSV file.

    Args:
        citations_df (pyspark.sql.DataFrame): The DataFrame containing the citations data.
        nlp_papers_df_with_categories (pyspark.sql.DataFrame): The DataFrame containing NLP papers with ARR categories.
        papers_df (pyspark.sql.DataFrame): The DataFrame containing the papers data.

    Returns:
        None
    """

    # Calculate citations from ARR papers to other categories
    arr_out_citations = (
        citations_df.join(
            nlp_papers_df_with_categories,
            citations_df.citingcorpusid
            == nlp_papers_df_with_categories.corpusid,
        )
        .join(papers_df, citations_df.citedcorpusid == papers_df.corpusid)
        .select(
            nlp_papers_df_with_categories["year"],
            nlp_papers_df_with_categories["arr_categories"],
            papers_df["s2fieldsofstudy"],
            "citedcorpusid",
            "citingcorpusid",
        )
    )

    arr_out_citations = (
        arr_out_citations.select(
            "year",
            explode("arr_categories").alias("arr_category"),
            "s2fieldsofstudy",
            "citedcorpusid",
            "citingcorpusid",
        )
        .select(
            "year",
            "arr_category",
            explode("s2fieldsofstudy").alias("s2fieldsofstudy"),
            "citedcorpusid",
            "citingcorpusid",
        )
        .select(
            "year",
            "arr_category",
            col("s2fieldsofstudy.category").alias("category"),
            "citedcorpusid",
            "citingcorpusid",
        )
    )

    citations2to1 = arr_out_citations.groupby(
        "year", "category", "arr_category"
    ).agg(
        count("citedcorpusid").alias("#citations1->2"),
        countDistinct("citingcorpusid").alias(
            "#papersfield1"
        ),  # Count distinct citing papers
    )

    # Calculate citations from other categories to ARR papers
    arr_in_citations = (
        citations_df.join(
            nlp_papers_df_with_categories,
            citations_df.citedcorpusid
            == nlp_papers_df_with_categories.corpusid,
        )
        .join(papers_df, citations_df.citingcorpusid == papers_df.corpusid)
        .select(
            nlp_papers_df_with_categories["year"],
            nlp_papers_df_with_categories["arr_categories"],
            papers_df["s2fieldsofstudy"],
            "citedcorpusid",
            "citingcorpusid",
        )
    )

    arr_in_citations = (
        arr_in_citations.select(
            "year",
            explode("arr_categories").alias("arr_category"),
            "s2fieldsofstudy",
            "citedcorpusid",
            "citingcorpusid",
        )
        .select(
            "year",
            "arr_category",
            explode("s2fieldsofstudy").alias("s2fieldsofstudy"),
            "citedcorpusid",
            "citingcorpusid",
        )
        .select(
            "year",
            "arr_category",
            col("s2fieldsofstudy.category").alias("category"),
            "citedcorpusid",
            "citingcorpusid",
        )
    )

    citations1to2 = arr_in_citations.groupby(
        "year", "category", "arr_category"
    ).agg(
        count("citedcorpusid").alias("#citations2->1"),
        countDistinct("citingcorpusid").alias(
            "#papersfield2"
        ),  # Count distinct cited papers
    )

    # Combine citation counts, paper counts, and ARR paper counts
    final_result = (
        citations1to2.join(
            citations2to1,
            on=["year", "category", "arr_category"],
            how="full_outer",
        )
        .withColumn("#citations1->2", coalesce(col("#citations1->2"), lit(0)))
        .withColumn("#citations2->1", coalesce(col("#citations2->1"), lit(0)))
        .withColumn("#papersfield1", coalesce(col("#papersfield1"), lit(0)))
        .withColumn("#papersfield2", coalesce(col("#papersfield2"), lit(0)))
        .select(
            col("year"),
            col("category").alias("field1"),
            col("arr_category").alias("field2"),
            col("#citations1->2"),
            col("#citations2->1"),
            col("#papersfield1"),
            col("#papersfield2"),
        )
    )

    # Order the result
    final_result = final_result.orderBy("year", "field1", ascending=False)

    # Convert to pandas
    final_result_pd = final_result.toPandas()

    # Save the result to a CSV file
    final_result_pd.to_csv(
        "./outputs/citations_arr_to_non_cs_fields_by_year.csv"
    )


def compute_nlp_subfield_to_nlp_subfield_citation_counts(
    nlp_papers_df_with_categories, citations_df
):
    """
    Computes citation counts from ARR papers to non-CS fields by year.

    The function performs the following steps:
    1. Joins the citations DataFrame with the nlp_papers_df_with_categories DataFrame to get the ARR papers' categories.
    2. Filters the joined DataFrame to include only ARR papers citing other ARR papers.
    3. Aggregates the citation counts and paper counts by year and field categories.
    4. Orders the result by year and field category in descending order.
    5. Converts the result to a Pandas DataFrame.
    6. Saves the result to a CSV file.

    Args:
        nlp_papers_df_with_categories (pyspark.sql.DataFrame): The DataFrame containing NLP papers with ARR categories.
        citations_df (pyspark.sql.DataFrame): The DataFrame containing the citations data.

    Returns:
        None
    """
    # Calculate citations from ARR papers to other ARR papers
    arr_to_arr_citations = (
        citations_df.join(
            nlp_papers_df_with_categories.alias("citing_papers"),
            citations_df.citingcorpusid == col("citing_papers.corpusid"),
            "inner",
        )
        .join(
            nlp_papers_df_with_categories.alias(
                "cited_papers"
            ).withColumnRenamed("arr_categories", "cited_arr_categories"),
            citations_df.citedcorpusid == col("cited_papers.corpusid"),
            "inner",
        )
        .select(
            col("citing_papers.year"),
            col("citing_papers.arr_categories").alias("arr_categories_citing"),
            col("cited_arr_categories").alias("arr_categories_cited"),
            "citedcorpusid",
            "citingcorpusid",
        )
    )

    arr_to_arr_citations = arr_to_arr_citations.select(
        "year",
        explode("arr_categories_citing").alias("arr_category_citing"),
        "arr_categories_cited",
        "citedcorpusid",
        "citingcorpusid",
    ).select(
        "year",
        "arr_category_citing",
        explode("arr_categories_cited").alias("arr_category_cited"),
        "citedcorpusid",
        "citingcorpusid",
    )

    # Aggregate citation counts and paper counts
    arr_to_arr_counts = arr_to_arr_citations.groupBy(
        "year", "arr_category_citing", "arr_category_cited"
    ).agg(
        count("citedcorpusid").alias("#citations"),
        countDistinct("citingcorpusid").alias("#papers_citing"),
        countDistinct("citedcorpusid").alias("#papers_cited"),
    )

    # Order the result
    final_result = arr_to_arr_counts.orderBy(
        "year", "arr_category_citing", "arr_category_cited", ascending=False
    )

    # Convert to pandas
    final_result_pd = final_result.toPandas()

    # Save the result to a CSV file
    final_result_pd.to_csv("./outputs/citations_arr_to_arr_by_year.csv")


def compute_subfields(nlp_papers_df):
    """
    Computes subfields for NLP papers based on the provided DataFrame.

    The function performs the following steps:
    1. Lowercases the titles in the DataFrame.
    2. Filters out null values in the title column.
    3. Removes punctuation from the titles.
    4. Tokenizes the titles into words.
    5. Removes stopwords from the words.
    6. Computes bigrams from the filtered words.
    7. Removes specific bigrams and non-English bigrams.
    8. Counts the occurrences of the bigrams.
    9. Extracts a regex pattern from the title column.
    10. Counts the occurrences of the regex pattern.
    11. Merges the bigram counts and regex counts.
    12. Finds the most frequent bigrams in the titles.
    13. Maps the bigrams to their ARR categories.
    14. Groups the papers by ID, title, and year, and aggregates the ARR categories into a list.
    15. Stores the result to a CSV file.
    16. Reads the bigrams-to-arr-categories CSV file.
    17. Joins the bigrams with the ARR categories in the DataFrame.
    18. Renames the "ARR Category" column to "arr_category".
    19. Groups the papers by ID, title, and year, and collects the ARR categories into a list.
    20. Shows the updated DataFrame with the arr_categories column.
    21. Counts the number of papers with non-empty ARR Category arrays.
    22. Stores the DataFrame to a CSV file.
    23. Returns the DataFrame with the computed subfields.

    Args:
        nlp_papers_df (pyspark.sql.DataFrame): The DataFrame containing the NLP papers data.

    Returns:
        pyspark.sql.DataFrame: The DataFrame with the computed subfields.
    """
    bigrams_file = "./outputs/bigram_counts.csv"

    # Lowercase titles
    nlp_papers_df_filtered = nlp_papers_df.withColumn(
        "title", lower(col("title"))
    )

    # Filter out null values
    nlp_papers_df_filtered = nlp_papers_df_filtered.na.drop(subset=["title"])

    # Remove punctuation
    regex = "[\\[\\]:;,\"'<>=~!%$@\\?\\.]"
    nlp_papers_df_filtered = nlp_papers_df_filtered.withColumn(
        "title", regexp_replace(col("title"), regex, "")
    )

    # Tokenize titles
    tokenizer = Tokenizer(inputCol="title", outputCol="words")
    words_df = tokenizer.transform(nlp_papers_df_filtered)

    # Remove stopwords
    stopwords = StopWordsRemover.loadDefaultStopWords("english")
    stopwords.extend(["based"])
    remover = StopWordsRemover(
        inputCol="words", outputCol="filtered_words", stopWords=stopwords
    )
    words_df = remover.transform(words_df)

    # Compute bigrams
    ngram = NGram(n=2, inputCol="filtered_words", outputCol="bigrams")
    bigrams_df = ngram.transform(words_df)

    # Add the bigram_str column back by joining the two bigrams with a space
    bigrams_df = bigrams_df.withColumn(
        "bigram_str", array_join(col("bigrams"), " ")
    )

    # Remove bigrams of form "task ##" and "task #"
    bigrams_df = bigrams_df.filter(
        ~col("bigram_str").rlike("task\\s\\d{2}")
        & ~col("bigram_str").rlike("task\\s\\d")
    )

    # Remove french, spanish, and german bigrams using language detection
    def safe_detect(x):
        """
        Computes bigrams from the given DataFrame.

        The function uses the NGram transformer to compute bigrams from the "filtered_words" column of the input DataFrame. The resulting bigrams are stored in the "bigrams" column of the output DataFrame.

        Args:
            n (int): The number of words in each bigram.
            inputCol (str): The name of the input column containing the filtered words.
            outputCol (str): The name of the output column to store the computed bigrams.

        Returns:
            pyspark.sql.DataFrame: The DataFrame with the computed bigrams in the "bigrams" column.
        """

        try:
            return detect(x)
        except LangDetectException:
            return "unknown"

    # Filter languages that are not english
    detect_udf = udf(safe_detect, StringType())
    bigrams_df = bigrams_df.withColumn(
        "language", detect_udf(col("bigram_str"))
    )
    bigrams_df = bigrams_df.filter(~col("language").isin(["fr", "es", "de"]))

    # Count bigram occurrences
    bigram_counts_df = (
        bigrams_df.select(explode(col("bigrams")).alias("bigram"))
        .groupBy("bigram")
        .count()
    )

    # Sort bigrams by count
    bigram_counts_df = bigram_counts_df.sort("count", ascending=False)

    # Extract the regex from the title column
    regex = "semeval-\d\d\d\d"
    nlp_papers_df_filtered = nlp_papers_df_filtered.withColumn(
        "semeval_year", regexp_extract(col("title"), regex, 0)
    )

    # Count occurrences of the regex
    semeval_counts_df = (
        nlp_papers_df_filtered.filter(col("semeval_year") != "")
        .groupBy("semeval_year")
        .count()
    )

    # Merge bigram_counts_df and semeval_counts_df
    merged_counts_df = bigram_counts_df.union(semeval_counts_df)

    # Sort by count descending and bigram ascending
    merged_counts_df = merged_counts_df.orderBy(
        col("count").desc(), col("bigram")
    )

    # Define a UDF to find bigrams in the title
    def find_bigrams(title, bigrams):
        """
        Finds the matched bigrams in the given title.

        Args:
            title (str): The title to search for bigrams.
            bigrams (list): The list of bigrams to match against the title.

        Returns:
            list: The matched bigrams found in the title."""

        matched_bigrams = [bigram for bigram in bigrams if bigram in title]
        return matched_bigrams

    # Find the bigrams in paper titles
    find_bigrams_udf = udf(find_bigrams, ArrayType(StringType()))

    # Compute the most frequent bigrams
    top_bigrams = (
        merged_counts_df.select("bigram")
        .limit(200)
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    # Apply the UDF to the DataFrame
    nlp_papers_df_filtered = nlp_papers_df_filtered.withColumn(
        "bigrams_in_title",
        find_bigrams_udf(
            col("title"), array([lit(bigram) for bigram in top_bigrams])
        ),
    )

    # Store to csv
    merged_counts_df.toPandas().to_csv(bigrams_file, header=True)

    # Read the bigrams-to-arr-categories.csv file
    bigrams_to_categories = spark.read.csv(
        "outputs/bigrams_to_arr_categories.csv", header=True
    )

    # Select only the relevant columns
    bigrams_to_categories = bigrams_to_categories.select(
        "Bigram", "ARR Category"
    )

    # Map the bigrams in the nlp_papers_df_filtered DataFrame to their ARR categories
    nlp_papers_df_with_categories = nlp_papers_df_filtered.alias(
        "papers"
    ).join(
        bigrams_to_categories.alias("categories"),
        array_contains(
            col("papers.bigrams_in_title"), col("categories.Bigram")
        ),
        how="left",
    )

    # Rename the "ARR Category" column to "arr_category"
    nlp_papers_df_with_categories = (
        nlp_papers_df_with_categories.withColumnRenamed(
            "ARR Category", "arr_category"
        )
    )

    # Group by paper ID and aggregate the categories into a list
    nlp_papers_df_with_categories = nlp_papers_df_with_categories.groupBy(
        "papers.corpusid", "papers.title", "papers.year"
    ).agg(collect_list("arr_category").alias("arr_categories"))

    # Show the updated DataFrame with the arr_categories column
    nlp_papers_df_with_categories.show()

    # Count the number of papers with non-empty ARR Category arrays
    non_empty_arr_category_count = nlp_papers_df_with_categories.filter(
        size(col("arr_categories")) > 0
    ).count()

    # Print the count
    print(
        "Number of papers with non-empty ARR Category arrays:"
        f" {non_empty_arr_category_count}"
    )

    nlp_papers_df_with_nlp_subfields_pd = (
        nlp_papers_df_with_categories.toPandas()
    )
    nlp_papers_df_with_nlp_subfields_pd.to_csv(
        "outputs/nlp_papers_to_nlp_subfields.csv"
    )

    return nlp_papers_df_with_categories


def compute_self_citations(papers_df, nlp_papers_df, citations_df):
    """
    Computes self-citation statistics based on the provided papers, NLP papers, and citations data.

    The function calculates self-citation percentages for NLP papers and papers in different fields of study. It performs the following steps:
    1. Joins NLP papers with the citations dataframe on the citingcorpusid.
    2. Calculates the total number of citations for NLP papers by year.
    3. Filters internal NLP citations by joining NLP papers on both the citingcorpusid and citedcorpusid.
    4. Calculates the number of internal citations for NLP papers by year.
    5. Joins total and internal citations counts by year.
    6. Calculates the self-citation percentage as the ratio of internal citations to total citations.
    7. Converts the result to a Pandas DataFrame and saves it as a CSV file.
    8. Performs the same steps for papers in different fields of study.
    9. Computes the macro-average self-citation percentage by averaging the self-citation percentage across all fields.
    10. Saves the macro-average self-citation DataFrame to a CSV file.

    Args:
        papers_df (pyspark.sql.DataFrame): The DataFrame containing the papers data.
        nlp_papers_df (pyspark.sql.DataFrame): The DataFrame containing the NLP papers data.
        citations_df (pyspark.sql.DataFrame): The DataFrame containing the citations data.

    Returns:
        None"""

    # Join NLP papers with the citations dataframe on the citingcorpusid
    all_citations_from_nlp = citations_df.join(
        nlp_papers_df,
        citations_df.citingcorpusid == nlp_papers_df.corpusid,
        "inner",
    )

    # Calculate the total number of citations for NLP papers
    total_citations_by_year = (
        all_citations_from_nlp.groupBy(col("year").alias("citation_year"))
        .count()
        .withColumnRenamed("count", "total_citations")
    )

    # Filter internal NLP citations by joining NLP papers on both the citingcorpusid and citedcorpusid
    internal_nlp_citations = all_citations_from_nlp.alias(
        "all_citations_from_nlp"
    ).join(
        nlp_papers_df.alias("nlp_papers_df_cited"),
        col("all_citations_from_nlp.citedcorpusid")
        == col("nlp_papers_df_cited.corpusid"),
        "inner",
    )

    internal_nlp_citations = internal_nlp_citations.join(
        nlp_papers_df.alias("nlp_papers_df_citing"),
        col("all_citations_from_nlp.citingcorpusid")
        == col("nlp_papers_df_citing.corpusid"),
        "inner",
    )

    # Calculate the number of internal citations for NLP papers
    internal_citations_by_year = (
        internal_nlp_citations.groupBy(
            col("nlp_papers_df_citing.year").alias("citation_year")
        )
        .count()
        .withColumnRenamed("count", "internal_citations")
    )

    # Join total and internal citations counts by year
    insularity_by_year = total_citations_by_year.join(
        internal_citations_by_year, "citation_year"
    )

    # Calculate insularity as the ratio of internal citations to total citations
    insularity_by_year = insularity_by_year.withColumn(
        "self_citation_percentage",
        col("internal_citations") / col("total_citations"),
    )

    # Convert to Pandas DataFrame for plotting
    insularity_by_year_pd = insularity_by_year.toPandas()
    insularity_by_year_pd.to_csv("outputs/insularity_NLP.csv")

    # Define a list of fields of study
    fields_of_study = ["Linguistics", "Psychology", "Mathematics"]

    # Initialize an empty dictionary to store the results
    insularity_by_year_all = {}

    for field in fields_of_study:
        # Filter papers to only include papers in the current field of study
        papers_in_field_df = papers_df.filter(
            array_contains(col("s2fieldsofstudy.category"), field)
        )

        # Join papers in the field with the citations dataframe on the citingcorpusid
        all_citations_from_field = citations_df.join(
            papers_in_field_df,
            citations_df.citingcorpusid == papers_in_field_df.corpusid,
            "inner",
        ).select(
            citations_df["*"],
            papers_in_field_df["year"].alias("citation_year"),
        )

        total_citations_by_year = (
            all_citations_from_field.groupBy("citation_year")
            .count()
            .withColumnRenamed("count", "total_citations")
        )

        internal_field_citations = all_citations_from_field.alias(
            "all_citations_from_field"
        ).join(
            papers_in_field_df.alias("papers_in_field_df_cited"),
            col("all_citations_from_field.citedcorpusid")
            == col("papers_in_field_df_cited.corpusid"),
            "inner",
        )

        internal_citations_by_year = (
            internal_field_citations.groupBy("citation_year")
            .count()
            .withColumnRenamed("count", "internal_citations")
        )

        # Join total and internal citations counts by year
        insularity_by_year = total_citations_by_year.join(
            internal_citations_by_year, "citation_year"
        )

        # Calculate insularity as the ratio of internal citations to total citations
        insularity_by_year = insularity_by_year.withColumn(
            "self_citation_percentage",
            col("internal_citations") / col("total_citations"),
        )

        # Convert to Pandas DataFrame for further processing and save the DataFrame to the dictionary
        insularity_by_year_all[field] = insularity_by_year.toPandas()

    # Save all DataFrames to CSV files
    for field, df in insularity_by_year_all.items():
        df.to_csv(f"outputs/insularity_{field}.csv")

    # Get all unique fields of study
    fields_of_study = (
        papers_df.select(explode(col("s2fieldsofstudy.category")))
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    # Initialize an empty dictionary to store the results
    insularity_by_year_all = {}

    for field in fields_of_study:
        # Filter papers to only include papers in the current field of study
        papers_in_field_df = papers_df.filter(
            array_contains(col("s2fieldsofstudy.category"), field)
        )

        # Take a 20% sample of the papers in this field for each year
        papers_in_field_df = papers_in_field_df.sampleBy(
            "year",
            fractions={
                y: 0.2
                for y in papers_in_field_df.select("year")
                .distinct()
                .rdd.flatMap(lambda x: x)
                .collect()
            },
            seed=42,
        )

        # Join papers in the field with the citations dataframe on the citingcorpusid
        all_citations_from_field = citations_df.join(
            papers_in_field_df,
            citations_df.citingcorpusid == papers_in_field_df.corpusid,
            "inner",
        ).select(
            citations_df["*"],
            papers_in_field_df["year"].alias("citation_year"),
        )

        total_citations_by_year = (
            all_citations_from_field.groupBy("citation_year")
            .count()
            .withColumnRenamed("count", "total_citations")
        )

        internal_field_citations = all_citations_from_field.alias(
            "all_citations_from_field"
        ).join(
            papers_in_field_df.alias("papers_in_field_df_cited"),
            col("all_citations_from_field.citedcorpusid")
            == col("papers_in_field_df_cited.corpusid"),
            "inner",
        )

        internal_citations_by_year = (
            internal_field_citations.groupBy("citation_year")
            .count()
            .withColumnRenamed("count", "internal_citations")
        )

        # Join total and internal citations counts by year
        insularity_by_year = total_citations_by_year.join(
            internal_citations_by_year, "citation_year"
        )

        # Calculate insularity as the ratio of internal citations to total citations
        insularity_by_year = insularity_by_year.withColumn(
            "self_citation_percentage",
            col("internal_citations") / col("total_citations"),
        )

        # Convert to Pandas DataFrame for further processing and save the DataFrame to the dictionary
        insularity_by_year_all[field] = insularity_by_year.toPandas()

    # Compute the macro-average self-citation percentage by averaging the self-citation percentage across all fields
    macro_avg_self_citation_df = (
        pd.concat(insularity_by_year_all.values())
        .groupby("citation_year")
        .mean()
        .reset_index()
    )

    # Save the DataFrame to a CSV file
    macro_avg_self_citation_df.to_csv("outputs/macro_avg_self_citation.csv")


def comput_general_stats(
    papers_df,
    nlp_papers_df,
    citations_df,
    overwrite=False,
    path="./outputs/general_stats.txt",
):
    """
    Computes general statistics based on the provided papers, NLP papers, and citations data.

    The function calculates various statistics, including the total number of papers, citations, NLP papers, citations from NLP papers, citations to NLP papers, citations from NLP papers to NLP papers, citations to NLP papers from non-NLP papers, and citations from NLP papers to non-NLP papers. The results are printed to the console and saved to a text file.

    Args:
        papers_df (pyspark.sql.DataFrame): The DataFrame containing the papers data.
        nlp_papers_df (pyspark.sql.DataFrame): The DataFrame containing the NLP papers data.
        citations_df (pyspark.sql.DataFrame): The DataFrame containing the citations data.
        overwrite (bool, optional): Whether to overwrite the existing general stats file if it exists. Defaults to False.
        path (str, optional): The path to save the general stats file. Defaults to "./outputs/general_stats.txt".

    Returns:
        None
    """
    if os.path.exists(general_stats_output_file) and not overwrite:
        print("General stats already computed, skipping...")
    else:
        c = citations_df.alias("c")
        n = nlp_papers_df.alias("n")

        num_papers = papers_df.count()
        print("Total number of papers:", num_papers)

        num_citations = c.count()
        print("Total number of citations:", num_citations)

        num_nlp_papers = n.count()
        print("Total number of NLP papers:", num_nlp_papers)

        nlp_citations_df = c.join(
            n.select("corpusid"), col("c.citingcorpusid") == col("n.corpusid")
        )
        num_nlp_citations = nlp_citations_df.count()
        print("Total number of citations from NLP papers:", num_nlp_citations)

        nlp_citations_to_df = c.join(
            n.select("corpusid"), col("c.citedcorpusid") == col("n.corpusid")
        )
        num_nlp_citations_to = nlp_citations_to_df.count()
        print("Total number of citations to NLP papers:", num_nlp_citations_to)

        nlp_citations_from_nlp_df = nlp_citations_df.join(
            n.select("corpusid").alias("nn"),
            col("c.citedcorpusid") == col("nn.corpusid"),
        )
        num_nlp_citations_from_nlp = nlp_citations_from_nlp_df.count()
        print(
            "Total number of citations to NLP papers from NLP papers:",
            num_nlp_citations_from_nlp,
        )

        non_nlp_papers_df = (
            c.join(
                n.select("corpusid"),
                col("c.citedcorpusid") == col("n.corpusid"),
                "leftanti",
            )
            .select("c.citedcorpusid")
            .distinct()
        )
        c_alias = c.alias("c_alias")
        nlp_citations_to_non_nlp_df = c_alias.join(
            non_nlp_papers_df,
            col("c_alias.citedcorpusid") == col("c.citedcorpusid"),
        ).join(
            n.select("corpusid"),
            col("c_alias.citingcorpusid") == col("n.corpusid"),
        )
        num_nlp_citations_from_non_nlp = nlp_citations_to_non_nlp_df.count()
        print(
            "Total number of citations to NLP papers from non-NLP papers:",
            num_nlp_citations_from_non_nlp,
        )

        with open(general_stats_output_file, "w") as f:
            f.write(f"Total number of papers: {num_papers}{os.linesep}")
            f.write(f"Total number of citations: {num_citations}{os.linesep}")
            f.write(
                f"Total number of NLP papers: {num_nlp_papers}{os.linesep}"
            )
            f.write(
                "Total number of citations from NLP papers:"
                f" {num_nlp_citations}{os.linesep}"
            )
            f.write(
                "Total number of citations to NLP papers:"
                f" {num_nlp_citations_to}{os.linesep}"
            )
            f.write(
                "Total number of citations from NLP papers to NLP papers:"
                f" {num_nlp_citations_from_nlp}{os.linesep}"
            )
            f.write(
                "Total number of citations to NLP papers from non-NLP papers:"
                f" {nlp_citations_to_non_nlp_df}{os.linesep}"
            )
            f.write(
                "Total number of citations from NLP papers to non-NLP papers:"
                f" {num_nlp_citations_from_non_nlp}{os.linesep}"
            )


def filter_s2fieldsofstudy(s2fieldsofstudy):
    """
    Computes rankings and citation analysis by epoch for a specified field.

    Args:
        field (str, optional): The field for which to compute the rankings and citation analysis. Defaults to "NLP".

    Returns:
        None

    Example:
        ```python
        compute_rankings_by_epoch("NLP")
        ```"""

    if s2fieldsofstudy is None:
        return None

    filtered_s2fieldsofstudy = []
    for field in s2fieldsofstudy:
        if field is None:
            continue
        if field["source"] == "external" and any(
            internal_field["category"] == field["category"]
            and internal_field["source"] != "external"
            for internal_field in s2fieldsofstudy
            if internal_field is not None
        ):
            continue
        if field not in filtered_s2fieldsofstudy:
            filtered_s2fieldsofstudy.append(field)
    return filtered_s2fieldsofstudy


def count_papers_per_field(papers_df):
    """
    Counts the number of papers per field of study.

    The function takes a DataFrame containing papers data and performs the following steps:
    1. Explode the s2fieldsofstudy column to create a new row for each field of study.
    2. Group the data by the category field and count the number of occurrences.
    3. Order the results by count in descending order.
    4. Convert the result to a Pandas DataFrame.
    5. Save the result as a CSV file.

    Args:
        papers_df (pyspark.sql.DataFrame): The DataFrame containing the papers data.

    Returns:
        None"""

    exploded_df = papers_df.select("*", explode("s2fieldsofstudy").alias("s2"))
    category_counts = (
        exploded_df.groupBy("s2.category")
        .count()
        .orderBy("count", ascending=False)
    )
    category_counts_pd = category_counts.toPandas()
    category_counts_pd.to_csv("outputs/general_stats_papers_per_field.csv")


def main():
    """
    Executes the main analysis process.

    The function performs various analysis tasks on the loaded papers and citations data. It includes the following steps:
    1. Load the papers and citations data.
    2. Apply a user-defined function to filter the s2fieldsofstudy column in the papers DataFrame.
    3. Filter the papers DataFrame to include only rows with non-null ACL externalids.
    4. Perform several analysis functions on the filtered papers and citations data.

    Args:
        None

    Returns:
        None"""
    overwrite = True
    sns.set(style="whitegrid")
    Path("./figures").mkdir(parents=True, exist_ok=True)
    Path("./outputs").mkdir(parents=True, exist_ok=True)

    # Load data
    papers_df = load_papers_data()
    citations_df = load_citations_data()

    filter_s2fieldsofstudy_udf = udf(
        filter_s2fieldsofstudy,
        ArrayType(
            StructType(
                [
                    StructField("category", StringType(), True),
                    StructField("source", StringType(), True),
                ]
            )
        ),
    )

    papers_df = papers_df.withColumn(
        "s2fieldsofstudy",
        filter_s2fieldsofstudy_udf(papers_df.s2fieldsofstudy),
    )
    nlp_papers_df = papers_df.filter(col("externalids.ACL").isNotNull())

    # Main analysis functions
    count_papers_per_field(papers_df)
    comput_general_stats(
        papers_df, nlp_papers_df, citations_df, overwrite=overwrite
    )
    nlp_papers_with_arr_cateogires = compute_subfields(nlp_papers_df)
    compute_nlp_subfield_to_nlp_subfield_citation_counts(
        nlp_papers_df_with_categories=nlp_papers_with_arr_cateogires,
        citations_df=citations_df,
    )
    compute_nlp_subfields_to_non_cs_fields_citation_counts(
        citations_df=citations_df,
        nlp_papers_df_with_categories=nlp_papers_with_arr_cateogires,
        papers_df=papers_df,
    )
    compute_non_cs_fields_and_nlp_citation_counts(
        nlp_papers_df=nlp_papers_df,
        citations_df=citations_df,
        papers_df=papers_df,
    )
    compute_non_cs_fields_and_non_cs_fields_citation_counts(
        papers_df=papers_df, citations_df=citations_df
    )
    compute_cfdi_non_cs_fields(papers_df=papers_df, citations_df=citations_df)
    compute_num_fields_per_paper(
        nlp_papers_df=nlp_papers_df,
        citations_df=citations_df,
    )


if __name__ == "__main__":
    main()
