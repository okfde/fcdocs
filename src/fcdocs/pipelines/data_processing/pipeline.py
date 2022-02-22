from kedro.pipeline import Pipeline, node

from .nodes import get_text_and_meta, sum_file_sizes


def create_pipeline(**kwargs):
    return Pipeline(
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
        ]
    )
