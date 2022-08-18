from kedro.pipeline import Pipeline, node, pipeline

from .nodes import calculate_features, get_text_and_meta, sum_file_sizes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_text_and_meta,
                inputs=["pdfdocuments", "params:max_workers", "params:spacy_model"],
                outputs="text_and_meta_dataframe",
                name="get_text_and_meta",
            ),
            node(
                func=calculate_features,
                inputs="text_and_meta_dataframe",
                outputs="data_with_features",
                name="calculate_features",
            ),
            node(
                func=sum_file_sizes,
                inputs="text_and_meta_dataframe",
                outputs="sum_of_filesizes",
                name="sum_file_sizes",
            ),
        ],
        namespace="data_processing",
        inputs="pdfdocuments",
        outputs=["data_with_features"],
    )
