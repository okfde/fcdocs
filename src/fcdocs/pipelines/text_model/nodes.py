"""
This is a boilerplate pipeline 'text_model'
generated using Kedro 0.18.0
"""
import logging

import pandas as pd
from sklearn.metrics import f1_score

from . import models


def get_model(model_class: str, model_args: dict):
    model = getattr(models, model_class)(**model_args)
    return model, {"class": model_class, "args": model_args}


def extract_x_y(data: pd.DataFrame):
    return extract_X(data), extract_y(data)


def extract_X(text_and_meta_dataframe: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(
        index=text_and_meta_dataframe.index,
    )
    features["id"] = text_and_meta_dataframe["id"]
    features["text"] = text_and_meta_dataframe["text"]
    return features


def extract_y(data: pd.DataFrame) -> pd.Series:
    return data.is_redacted


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has an F1 score of %.3f on test data.", score)
    return {"f1_score": score}


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    return model.fit(X_train, y_train)
