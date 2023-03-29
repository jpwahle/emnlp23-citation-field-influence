from data import (
    get_spark_session,
    load_citations_data_original,
    load_papers_data_original,
    write_citations_data,
    write_papers_data,
)


def main():
    """
    Executes the main preprocessing logic.

    This function loads the original papers and citations data using Spark, selects the required columns, and writes the processed data to the appropriate locations.

    Args:
        None

    Returns:
        None"""

    spark = get_spark_session()

    papers_df = load_papers_data_original(spark)
    citations_df = load_citations_data_original(spark)

    papers_df = papers_df.select(
        "corpusid",
        "externalids",
        "title",
        "authors",
        "venue",
        "publicationvenueid",
        "year",
        "referencecount",
        "citationcount",
        "influentialcitationcount",
        "isopenaccess",
        "s2fieldsofstudy",
    )
    citations_df = citations_df.select(
        "citingcorpusid", "citedcorpusid", "contexts", "intents"
    )

    write_papers_data(papers_df)
    write_citations_data(citations_df)


if __name__ == "__main__":
    main()
