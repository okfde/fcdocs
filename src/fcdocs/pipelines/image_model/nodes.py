import logging

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from . import models


def get_model(model_class: str, model_args: dict):
    model = getattr(models, model_class)(**model_args)
    return model, {"class": model_class, "args": model_args}


def extract_x_y(data: pd.DataFrame):
    return extract_X(data), extract_y(data)


def extract_X(data: pd.DataFrame) -> pd.DataFrame:
    return data[["id", "dark_ratio"]]


def extract_y(data: pd.DataFrame) -> pd.Series:
    return data.is_redacted.apply(float)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    scores = {
        "f1_score": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    logger = logging.getLogger(__name__)
    logger.info(f"Model score: {scores}")
    return scores


def train_model(untrained_model, X_train: pd.DataFrame, y_train: pd.Series):
    return untrained_model.fit(X_train, y_train)
