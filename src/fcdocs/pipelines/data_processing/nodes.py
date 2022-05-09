import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Tuple

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

    def load_single(arg):
        _partition_key, partition_load_func = arg
        docdata: DocumentData = partition_load_func()
        meta = docdata.meta
        meta["text"] = docdata.text
        meta["image"] = docdata.image
        return meta

    def load():
        with ThreadPoolExecutor() as executor:
            return executor.map(
                load_single,
                sorted(partitioned_input.items()),
            )

    df = pd.DataFrame(load())
    logger.info("DTypes of dataframe: \n%s", df.dtypes)
    return df


def split_data(
    text_and_meta_dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_train = text_and_meta_dataframe.sample(frac=0.9, random_state=0)
    data_test = text_and_meta_dataframe.drop(data_train.index)
    return data_train, data_test


def sum_file_sizes(text_and_meta_dataframe: pd.DataFrame) -> int:
    sum_of_filesizes = text_and_meta_dataframe["size"].sum()
    logger.info("Combined file size: %s Bytes", sum_of_filesizes)
    return sum_of_filesizes
