"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import data_processing as dp
from .pipelines import image_model as im
from .pipelines import text_model as tm


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
    A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dp.create_pipeline()
    text_model_pipeline = tm.create_pipeline()
    image_model_pipeline = im.create_pipeline()

    return {
        "__default__": data_processing_pipeline
        + text_model_pipeline
        + image_model_pipeline,
        "dp": data_processing_pipeline,
        "tm": text_model_pipeline,
        "im": image_model_pipeline,
    }
