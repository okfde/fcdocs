import logging
from typing import Dict, List, Tuple, Union

import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

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


def extract_data(data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    data = data[data.text.str.len() > 0]
    return data[features]


def evaluate_model(model, X_test: pd.DataFrame) -> ModelScore:
    y_pred = model.predict(X_test)
    x_test = model.process(X_test)
    logging.info(f"prediction={y_pred}")
    scores = {
        "calinski_harabasz_score": calinski_harabasz_score(x_test, y_pred),
        "davies_bouldin_score": -davies_bouldin_score(x_test, y_pred),
    }
    logger = logging.getLogger(__name__)
    logger.info(f"Model score: {scores}")
    return scores


def evaluate_models(models, X_test: pd.DataFrame) -> List[Dict[str, float]]:
    scores = [evaluate_model(model, X_test) for model in models]
    return scores


def train_models(models, X_train: pd.DataFrame):
    trained_models = []
    for model in models:
        trained_models.append(model.fit(X_train))
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
    X: pd.DataFrame, train_percentage: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    X_train = X.sample(frac=train_percentage / 100, random_state=0)
    X_dev = X.drop(X_train.index).sample(frac=0.5, random_state=0)
    X_test = X.drop(X_train.index).drop(X_dev.index)
    logger = logging.getLogger(__name__)
    logger.info(f"train/dev/test: {len(X_train)}/{len(X_dev)}/{len(X_test)}")
    return X_train, X_dev, X_test
