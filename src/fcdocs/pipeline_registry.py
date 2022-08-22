"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline

from .pipelines import classifier as tm
from .pipelines import data_processing as dp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
    A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dp.create_pipeline()
    classifier_pipeline = pipeline(
        tm.create_pipeline(),
        inputs={
            "model_config": "params:classifier.model_config",
        },
        outputs={"best_model": "classifier.model"},
    )
    return {
        "__default__": data_processing_pipeline + classifier_pipeline,
        "dp": data_processing_pipeline,
        "tm": classifier_pipeline,
    }
