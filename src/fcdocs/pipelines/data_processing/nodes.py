import logging
from typing import Any, Callable

import pandas as pd

from ...extras.datasets.document_dataset import DocumentData

logger = logging.getLogger(__name__)


def get_text_and_meta(partitioned_input: dict[str, Callable[[], Any]]) -> pd.DataFrame:
    """Concatenate input partitions into one pandas DataFrame.

    Args:
        partitioned_input: A dictionary with partition ids as keys and load functions as values.

    Returns:
        Pandas DataFrame representing a concatenation of all loaded partitions.
    """

    def load():
        for partition_key, partition_load_func in sorted(partitioned_input.items()):
            # load the PDF data
            docdata: DocumentData = partition_load_func()
            meta = docdata.meta
            meta["text"] = docdata.text
            yield meta

    df = pd.DataFrame(load())
    logger.info("DTypes of dataframe: \n%s", df.dtypes)
    return df


def sum_file_sizes(text_and_meta_dataframe: pd.DataFrame) -> int:
    sum_of_filesizes = text_and_meta_dataframe["size"].sum()
    logger.info("Combined file size: %s Bytes", sum_of_filesizes)
    return sum_of_filesizes
