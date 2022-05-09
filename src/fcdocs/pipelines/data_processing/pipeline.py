from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_text_and_meta, split_data, sum_file_sizes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_text_and_meta,
                inputs="pdfdocuments",
                outputs="text_and_meta_dataframe",
                name="get_text_and_meta",
            ),
            node(
                func=sum_file_sizes,
                inputs="text_and_meta_dataframe",
                outputs="sum_of_filesizes",
                name="sum_file_sizes",
            ),
            node(
                func=split_data,
                inputs="text_and_meta_dataframe",
                outputs=["data_train", "data_test"],
                name="split_data",
            ),
        ],
        namespace="data_processing",
        inputs="pdfdocuments",
        outputs=["data_train", "data_test"],
    )
