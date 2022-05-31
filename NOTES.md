# Research notes

## PDF tooling

[Good overview of PDF tooling](https://johannesfilter.com/python-and-pdf-a-review-of-existing-tools/)

## Possible Goals

- Document classification: which category does a document belong to?
- Document clustering: sort documents automatically into groups
- Document splitting: split a document up if they contain two different types of documents


## Possible approaches

Discussion with data science pracitioners yielded the following possible approaches.

- establishing a baseline classification, e.g. with simple word matching
- limit text extraction to first `X` pages of a document
- limit image-based features to first page of document

The classical route, based on domain knowledge:

- make it easy to create abstract features based on domain knowledge (e.g. "percent dark pixels in top third of first page") from 
- use standard statistical classifiers like Naive Bayes or Support Vector Classifiers

Especially for text classification, pre-trained text models like [fasttext](https://fasttext.cc/) can be used as an alternative to manual feature engineering.
