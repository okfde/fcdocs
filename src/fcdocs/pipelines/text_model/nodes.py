"""
This is a boilerplate pipeline 'text_model'
generated using Kedro 0.18.0
"""
import logging

import pandas as pd
from sklearn.metrics import f1_score

from .models import BaselineModel


def get_baseline_model():
    return BaselineModel()


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has an F1 score of %.3f on test data.", score)
    return {"f1_score": score}
