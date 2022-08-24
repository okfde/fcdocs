from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_model,
    evaluate_models,
    extract_data,
    get_models,
    select_best_model,
    split_data,
    train_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extract_data,
                inputs=[
                    "data_with_features",
                    "params:input_features",
                ],
                outputs="X",
                name="extract_data",
            ),
            node(
                func=split_data,
                inputs=["X", "params:train_percentage"],
                outputs=["X_train", "X_dev", "X_test"],
                name="split_data",
            ),
            node(
                func=get_models,
                inputs=["model_config"],
                outputs=["untrained_models", "params"],
                name="get_models",
            ),
            node(
                func=train_models,
                inputs=["untrained_models", "X_train"],
                outputs="models",
                name="train_models",
            ),
            node(
                func=evaluate_models,
                inputs=["models", "X_dev"],
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
                inputs=["best_model", "X_test"],
                outputs="score",
                name="evalute_best_model",
            ),
        ],
        inputs=["model_config", "data_with_features"],
        namespace="clustering",
        outputs=["best_model"],
    )
