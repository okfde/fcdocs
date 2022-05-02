"""
This is a boilerplate pipeline 'text_model'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, get_baseline_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_baseline_model,
                inputs=None,
                outputs="baseline_model",
                name="get_baseline_model",
            ),
            node(
                func=evaluate_model,
                inputs=["baseline_model", "X_test", "y_test"],
                outputs="predicted_test_dataframe",
                name="evaluate_baseline_model",
            ),
        ]
    )
