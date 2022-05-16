"""
This is a boilerplate pipeline 'text_model'
generated using Kedro 0.18.0
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, extract_x_y, get_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extract_x_y,
                inputs=[
                    "data_train",
                    "params:input_features",
                    "params:predict_feature",
                ],
                outputs=["X_train", "y_train"],
                name="extract_train",
            ),
            node(
                func=extract_x_y,
                inputs=[
                    "data_test",
                    "params:input_features",
                    "params:predict_feature",
                ],
                outputs=["X_test", "y_test"],
                name="extract_test",
            ),
            node(
                func=get_model,
                inputs=["model_class", "model_args"],
                outputs=["untrained_model", "params"],
                name="get_model",
            ),
            node(
                func=train_model,
                inputs=["untrained_model", "X_train", "y_train"],
                outputs="model",
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs="scores",
                name="evaluate_model",
            ),
        ],
        inputs=["model_class", "model_args", "data_train", "data_test"],
        namespace="text_model",
        outputs=["model"],
    )
