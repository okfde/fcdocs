"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline

from .pipelines import data_processing as dp
from .pipelines import image_model as im
from .pipelines import text_model as tm


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
    A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dp.create_pipeline()
    text_model_pipeline = pipeline(
        tm.create_pipeline(),
        inputs={
            "model_config": "params:text_model.model_config",
        },
        outputs={"best_model": "text_model.model"},
    )
    image_model_pipeline = pipeline(
        im.create_pipeline(),
        inputs={
            "model_class": "params:image_model.model_class",
            "model_args": "params:image_model.model_args",
        },
        outputs={"model": "image_model.model"},
    )

    return {
        "__default__": data_processing_pipeline
        + text_model_pipeline
        + image_model_pipeline,
        "dp": data_processing_pipeline,
        "tm": text_model_pipeline,
        "im": image_model_pipeline,
    }
