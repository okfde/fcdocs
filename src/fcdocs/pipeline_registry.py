"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline

from .pipelines import classifier as cf
from .pipelines import clustering as cs
from .pipelines import data_processing as dp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
    A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dp.create_pipeline()
    classifier_pipeline = pipeline(
        cf.create_pipeline(),
        inputs={
            "model_config": "params:classifier.model_config",
        },
        outputs={"best_model": "classifier.model"},
    )
    clustering_pipeline = pipeline(
        cs.create_pipeline(),
        inputs={
            "model_config": "params:clustering.model_config",
        },
        outputs={"best_model": "clustering.model"},
    )
    return {
        "__default__": data_processing_pipeline + classifier_pipeline,
        "dp": data_processing_pipeline,
        "cf": classifier_pipeline,
        "cs": clustering_pipeline,
    }
