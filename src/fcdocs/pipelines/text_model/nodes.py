"""
This is a boilerplate pipeline 'text_model'
generated using Kedro 0.18.0
"""
import logging
from typing import Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from . import models


def get_model(model_class: str, model_args: Dict):
    model = getattr(models, model_class)(**model_args)
    return model, {"class": model_class, "args": model_args}


def extract_x_y(data: pd.DataFrame, x_features: List[str], predict_feature: str):
    return extract_X(data, x_features), data[predict_feature].apply(float)


def extract_X(data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return data[features]


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
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


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    return model.fit(X_train, y_train)
