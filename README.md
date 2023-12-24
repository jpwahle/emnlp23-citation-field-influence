# We are Who We Cite: Bridges of Influence Between Natural Language Processing and Other Academic Fields
[![arXiv](https://img.shields.io/badge/arXiv-2310.14870-b31b1b.svg)](https://arxiv.org/abs/2310.14870)
[![HuggingFace Demo](https://img.shields.io/badge/🤗-Demo-ffce1c.svg)](https://huggingface.co/spaces/jpwahle/field-diversity)

## The Repository

This repository implements the EMNLP'23 paper "We are Who We Cite: Bridges of Influence Between Natural Language Processing and Other Academic Fields"

## Getting Started

### Data

First, you need  to download a recent dump from Semantic Scholar. Therefore, set your API key under `YOUR_API_KEY`, and request the following two endpoints:

```bash
curl --location 'https://api.semanticscholar.org/datasets/v1/release/2023-01-03/dataset/citations' \
--header 'x-api-key: ${YOUR_API_KEY}' \
-o citations/citations.json
```

```bash
curl --location 'https://api.semanticscholar.org/datasets/v1/release/2023-01-03/dataset/papers' \
--header 'x-api-key: ${YOUR_API_KEY}' \
-o papers/papers.json
```

Next, execute the following command to download the entire dataset.
> Note: This will take up significant space 108G+ for papers and 534G+ for citations compressed.

```bash
python3 src/download.py
```

### Pre-Processing

To convert the dataset into [Apache Spark's](https://spark.apache.org) native format which increases processing speeds significantly, execute:

```bash
python3 src/preprocess.py
```

### Analysis

To reproduce the analysis, execute:
>Note: Because some operations require filtering and joining millions of papers, they can take sometimes 24h+. You can choose which functions to run in the main() function.

```bash
python3 src/analysis.py
```

## Contributing

There are many ways in which you can participate in this project, for example:

* [Submit bugs and feature requests](https://github.com/jpwahle/emnlp23-citation-field-influence/issue), and help us verify as they are checked in
* Review [source code changes](https://github.com/jpwahle/emnlp23-citation-field-influence/pulls)

## Citation


```bib
@inproceedings{wahle-etal-2023-cite,
    title = "We are Who We Cite: Bridges of Influence Between Natural Language Processing and Other Academic Fields",
    author = "Wahle, Jan Philip  and
      Ruas, Terry  and
      Abdalla, Mohamed  and
      Gipp, Bela  and
      Mohammad, Saif",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.797",
    doi = "10.18653/v1/2023.emnlp-main.797",
    pages = "12896--12913",
    abstract = "Natural Language Processing (NLP) is poised to substantially influence the world. However, significant progress comes hand-in-hand with substantial risks. Addressing them requires broad engagement with various fields of study. Yet, little empirical work examines the state of such engagement (past or current). In this paper, we quantify the degree of influence between 23 fields of study and NLP (on each other). We analyzed {\textasciitilde}77k NLP papers, {\textasciitilde}3.1m citations from NLP papers to other papers, and {\textasciitilde}1.8m citations from other papers to NLP papers. We show that, unlike most fields, the cross-field engagement of NLP, measured by our proposed Citation Field Diversity Index (CFDI), has declined from 0.58 in 1980 to 0.31 in 2022 (an all-time low). In addition, we find that NLP has grown more insular{---}citing increasingly more NLP papers and having fewer papers that act as bridges between fields. NLP citations are dominated by computer science; Less than 8{\%} of NLP citations are to linguistics, and less than 3{\%} are to math and psychology. These findings underscore NLP{'}s urgent need to reflect on its engagement with various fields.",
}
```

Also make sure to cite the following paper if you use SemanticScholar data:

```bib
@inproceedings{lo-wang-2020-s2orc,
    title = "{S}2{ORC}: The Semantic Scholar Open Research Corpus",
    author = "Lo, Kyle  and Wang, Lucy Lu  and Neumann, Mark  and Kinney, Rodney  and Weld, Daniel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.447",
    doi = "10.18653/v1/2020.acl-main.447",
    pages = "4969--4983"
}
```

## License

Licensed under the [Apache 2.0](LICENSE.txt) license.

