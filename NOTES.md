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


### The classical route, based on domain knowledge:

- make it easy to create abstract features based on domain knowledge (e.g. "percent dark pixels in top third of first page") from 
- use standard statistical classifiers like Naive Bayes or Support Vector Classifiers

### Classical NLP

Classical lemmatization and Named Entity Recognition can extract entities from documents which can be used for clustering or classification.

### Deep learning models

Pre-trained word embedding models like [fasttext](https://fasttext.cc/) can be used as an alternative to manual feature engineering.

Key-value pair extraction from structured documents (like invoices). [Text+image models by Microsoft](https://github.com/microsoft/unilm), e.g. [for document layout tokenization and key-value pair extraction](https://huggingface.co/docs/transformers/main/model_doc/layoutxlm)


## Summary

We'd like to try a mix of these approaches to see how they perform on various classification problems. The modular data pipeline framework allows to separate and test them individually.
