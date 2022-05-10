import logging

import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from . import models


def dark_pixel_ratio(image: Image):
    dark_pixels = 0
    greyscale_image = image.convert("L")
    for color in greyscale_image.getdata():
        if color / 255 < 0.25:
            dark_pixels += 1

    total_pixels = greyscale_image.width * greyscale_image.height
    return dark_pixels / total_pixels


def extract_dark_ratio(
    text_and_meta_dataframe: pd.DataFrame,
) -> pd.Series:
    return text_and_meta_dataframe["image"].map(dark_pixel_ratio)


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
    features["dark_ratio"] = extract_dark_ratio(text_and_meta_dataframe).to_frame()
    return features


def extract_y(data: pd.DataFrame) -> pd.Series:
    return data.is_redacted


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
