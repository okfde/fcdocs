from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, extract_features, extract_y, get_baseline_model


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
                func=extract_y,
                inputs="data_test",
                outputs="y_test",
                name="extract_y_test",
            ),
            node(
                func=extract_features,
                inputs="data_test",
                outputs="X_test",
                name="extract_features",
            ),
            node(
                func=evaluate_model,
                inputs=["baseline_model", "X_test", "y_test"],
                outputs="predicted_test_dataframe",
                name="evaluate_baseline_model",
            ),
        ],
        inputs=["data_test"],
        namespace="image_model",
    )
