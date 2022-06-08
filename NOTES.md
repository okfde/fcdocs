# Research notes

## Existing tooling around document processing

- [Overview of PDF tooling](https://johannesfilter.com/python-and-pdf-a-review-of-existing-tools/)
- [Tesseract, the default for Open Source OCR](https://tesseract-ocr.github.io/tessdoc/)
- [Paddle a newer OCR solution](https://github.com/PaddlePaddle/PaddleOCR)

## Possible Goals

- Document classification: which category does a document belong to?
- Document clustering: sort documents automatically into groups
- Document splitting: split a document up if they contain two different types of documents


## Approach

Discussion with data science pracitioners yielded the following approaches.

- Use a standardised and documented data processing and analysis pipeline ([Kedro](https://kedro.readthedocs.io/))
- Bring documents into our pipeline and prepare them (extract text, generate thumbnails). Limit processing to `X` number of pages / smaller documents.
- Establish a baseline classification, e.g. with simple word matching to judge performance of more sophisticated approaches
- Build Webservice to annotate documents manually in order to generate testdata.


### The classical route, based on domain knowledge:

- make it easy to create abstract features based on domain knowledge (e.g. "percent dark pixels in top third of first page") from 
- use standard statistical classifiers like Naive Bayes or Support Vector Classifiers

### Classical NLP

Classical lemmatization and Named Entity Recognition can extract entities from documents which can be used for clustering or classification.

### Deep learning models

Pre-trained word embedding models like [fasttext](https://fasttext.cc/) can be used as an alternative to manual feature engineering.

Key-value pair extraction from structured documents (like invoices). [Text+image models by Microsoft](https://github.com/microsoft/unilm), e.g. [for document layout tokenization and key-value pair extraction](https://huggingface.co/docs/transformers/main/model_doc/layoutxlm)


## Summary and results

We'd like to try a mix of these approaches to see how they perform on various classification problems. 

To set up our data cleaning and exploration pipeline we decided to use the open source framework Kedro. It makes reproducible pipelines and model generation easier and more manageable than a collection of Jupyter notebooks.

The code is published on Github: [https://github.com/okfde/fcdocs](https://github.com/okfde/fcdocs)

We decided to develop a web application to create the test data needed for the subsequent machine learning. The basic functionality is that documents (PDFs) can be loaded into the application and then annotated by users with certain properties (such as 'this document is redacted' or 'this document is a letter'). The queried properties are configurable via a web interface. The results can be retrieved via an API and can be integrated into the further process.

The code is published on Github: [https://github.com/okfde/fcdocs-annotate](https://github.com/okfde/fcdocs-annotate)  
In addition, the application can be tried out here: [https://fragdenstaat.de/fcdocs_annotate/](https://fragdenstaat.de/fcdocs_annotate/)
