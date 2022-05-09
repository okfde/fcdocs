# Pipeline image_model
## Overview

This pipeline takes the pre-processed data from the `data_processing` pipeline
and trains an imaged based model with it.

If then evaluates this model on the test data and reports the f1 score.

## Pipeline inputs

The pipeline needs the `data_train` and `data_test` inputs from the
`data_processing` pipeline.

It also needs to know which model to train. You can set this in the
`conf/parameters.yml` file using the `image_model.model_class` parameter.
## Pipeline outputs

The pipeline outputs the trained model in the `model` variable.
