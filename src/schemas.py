from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def get_original_papers_schema():
    """
    Returns the schema for the original papers.

    Returns:
        pyspark.sql.types.StructType: The schema for the original papers.

    Example:
        ```python
        schema = get_original_papers_schema()
        print(schema)
        ```"""
    return StructType(
        [
            StructField("corpusid", IntegerType(), True),
            StructField(
                "externalids",
                StructType(
                    [
                        StructField("ACL", StringType(), True),
                        StructField("DBLP", StringType(), True),
                        StructField("ArXiv", StringType(), True),
                        StructField("MAG", StringType(), True),
                        StructField("CorpusId", StringType(), True),
                        StructField("PubMed", StringType(), True),
                        StructField("DOI", StringType(), True),
                        StructField("PubMedCentral", StringType(), True),
                    ]
                ),
                True,
            ),
            StructField("url", StringType(), True),
            StructField("title", StringType(), True),
            StructField(
                "authors",
                ArrayType(
                    StructType(
                        [
                            StructField("authorId", StringType(), True),
                            StructField("name", StringType(), True),
                        ]
                    )
                ),
                True,
            ),
            StructField("venue", StringType(), True),
            StructField("publicationvenueid", StringType(), True),
            StructField("year", IntegerType(), True),
            StructField("referencecount", IntegerType(), True),
            StructField("citationcount", IntegerType(), True),
            StructField("influentialcitationcount", IntegerType(), True),
            StructField("isopenaccess", BooleanType(), True),
            StructField(
                "s2fieldsofstudy",
                ArrayType(
                    StructType(
                        [
                            StructField("category", StringType(), True),
                            StructField("source", StringType(), True),
                        ]
                    )
                ),
                True,
            ),
            StructField("publicationtypes", ArrayType(StringType()), True),
            StructField("publicationdate", TimestampType(), True),
            StructField(
                "journal",
                StructType(
                    [
                        StructField("name", StringType(), True),
                        StructField("pages", StringType(), True),
                        StructField("volume", StringType(), True),
                    ]
                ),
                True,
            ),
            StructField("updated", TimestampType(), True),
        ]
    )


def get_truncated_papers_schema():
    """
    Returns the schema for the truncated papers (those with the minimum information required to perform the analysis).

    Returns:
        pyspark.sql.types.StructType: The schema for the truncated papers."""

    return StructType(
        [
            StructField("corpusid", IntegerType(), True),
            StructField(
                "externalids",
                StructType(
                    [
                        StructField("ACL", StringType(), True),
                        StructField("DBLP", StringType(), True),
                        StructField("ArXiv", StringType(), True),
                        StructField("MAG", StringType(), True),
                        StructField("CorpusId", StringType(), True),
                        StructField("PubMed", StringType(), True),
                        StructField("DOI", StringType(), True),
                        StructField("PubMedCentral", StringType(), True),
                    ]
                ),
                True,
            ),
            StructField("title", StringType(), True),
            StructField(
                "authors",
                ArrayType(
                    StructType(
                        [
                            StructField("authorId", StringType(), True),
                            StructField("name", StringType(), True),
                        ]
                    )
                ),
                True,
            ),
            StructField("venue", StringType(), True),
            StructField("publicationvenueid", StringType(), True),
            StructField("year", IntegerType(), True),
            StructField("referencecount", IntegerType(), True),
            StructField("citationcount", IntegerType(), True),
            StructField("influentialcitationcount", IntegerType(), True),
            StructField("isopenaccess", BooleanType(), True),
            StructField(
                "s2fieldsofstudy",
                ArrayType(
                    StructType(
                        [
                            StructField("category", StringType(), True),
                            StructField("source", StringType(), True),
                        ]
                    )
                ),
                True,
            ),
        ]
    )


def get_original_citations_schema():
    """
    Returns the schema for the original citations.

    Returns:
        pyspark.sql.types.StructType: The schema for the original citations."""

    return StructType(
        [
            StructField("citingcorpusid", StringType(), True),
            StructField("citedcorpusid", StringType(), True),
            StructField("isinfluential", BooleanType(), True),
            StructField("contexts", ArrayType(StringType()), True),
            StructField("intents", ArrayType(StringType()), True),
            StructField("updated", TimestampType(), True),
        ]
    )


def get_truncated_citations_schema():
    """
    Returns the schema for the truncated citations (those with the minimum information required to perform the analysis).

    Returns:
        pyspark.sql.types.StructType: The schema for the truncated citations.
    """

    return StructType(
        [
            StructField("citingcorpusid", StringType(), True),
            StructField("citedcorpusid", StringType(), True),
            StructField("contexts", ArrayType(StringType()), True),
            StructField("intents", ArrayType(StringType()), True),
        ]
    )
