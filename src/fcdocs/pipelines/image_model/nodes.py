import logging

import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

from .models import BaselineModel


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


def get_baseline_model():
    return BaselineModel()


def extract_features(text_and_meta_dataframe: pd.DataFrame) -> pd.DataFrame:
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
    score = f1_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has an F1 score of %.3f on test data.", score)
    return {"f1_score": score}
