"""
This is a boilerplate pipeline 'text_model'
generated using Kedro 0.18.0
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_model,
    evaluate_models,
    extract_x_y,
    get_models,
    select_best_model,
    train_models,
)


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
                    "data_dev",
                    "params:input_features",
                    "params:predict_feature",
                ],
                outputs=["X_dev", "y_dev"],
                name="extract_dev",
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
                func=get_models,
                inputs=["model_config"],
                outputs=["untrained_models", "params"],
                name="get_models",
            ),
            node(
                func=train_models,
                inputs=["untrained_models", "X_train", "y_train"],
                outputs="models",
                name="train_models",
            ),
            node(
                func=evaluate_models,
                inputs=["models", "X_dev", "y_dev"],
                outputs="scores",
                name="evaluate_models",
            ),
            node(
                func=select_best_model,
                inputs=["models", "scores", "params:selection_score"],
                outputs="best_model",
                name="select_best_model",
            ),
            node(
                func=evaluate_model,
                inputs=["best_model", "X_test", "y_test"],
                outputs="score",
                name="evalute_best_model",
            ),
        ],
        inputs=["model_config", "data_train", "data_test", "data_dev"],
        namespace="text_model",
        outputs=["best_model"],
    )
