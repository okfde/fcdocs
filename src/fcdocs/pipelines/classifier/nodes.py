import logging
from typing import Dict, List, Tuple, Union

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from . import models

ModelScore = Dict[str, float]


def get_models(model_config: List[Dict[str, Union[str, dict]]]):
    initialized_models = []
    model_params = []
    for model_config in model_config:
        model_class = model_config["class"]
        model_args = model_config.get("args", {})
        model = getattr(models, model_class)(**model_args)
        initialized_models.append(model)
        model_params.append({"class": model_class, "args": model_args})
    return initialized_models, model_params


def extract_x_y(data: pd.DataFrame, x_features: List[str], predict_feature: str):

    logger = logging.getLogger(__name__)
    logger.info("DTypes of dataframe: \n%s", data.dtypes)

    data = data[data[predict_feature].notna()]
    return extract_X(data, x_features), data[predict_feature].astype("bool")


def extract_X(data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return data[features]


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> ModelScore:
    y_pred, _ = model.predict(X_test)
    logging.info(f"gold={y_test}, prediction={y_pred}")
    scores = {
        "f1_score": f1_score(y_test, y_pred),
        "macro_f1_score": f1_score(y_test, y_pred, average="macro"),
        "micro_f1_score": f1_score(y_test, y_pred, average="micro"),
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
    }
    logger = logging.getLogger(__name__)
    logger.info(f"Model score: {scores}")
    return scores


def evaluate_models(
    models, X_test: pd.DataFrame, y_test: pd.Series
) -> List[Dict[str, float]]:
    scores = [evaluate_model(model, X_test, y_test) for model in models]
    return scores


def train_models(models, X_train: pd.DataFrame, y_train: pd.Series):
    trained_models = []
    for model in models:
        trained_models.append(model.fit(X_train, y_train))
    return trained_models


def select_best_model(models, scores: List[ModelScore], selection_score: str):
    best_score = None
    best_model = None
    for model, model_score in zip(models, scores):
        score = model_score[selection_score]
        if best_score is None or best_score < score:
            best_score = score
            best_model = model
    return best_model


def split_data(
    X: pd.DataFrame, y: pd.Series, train_percentage: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    X_train = X.sample(frac=train_percentage / 100, random_state=0)
    X_dev = X.drop(X_train.index).sample(frac=0.5, random_state=0)
    X_test = X.drop(X_train.index).drop(X_dev.index)
    y_train = y[X_train.index]
    y_dev = y[X_dev.index]
    y_test = y[X_test.index]
    logger = logging.getLogger(__name__)
    logger.info(f"train/dev/test: {len(X_train)}/{len(X_dev)}/{len(X_test)}")
    return X_train, y_train, X_dev, y_dev, X_test, y_test
