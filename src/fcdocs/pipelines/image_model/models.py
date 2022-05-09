import pandas as pd
from sklearn.linear_model import (  # noqa: F401
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
)


class BaselineModel:
    def fit(self, data: pd.DataFrame, targets: pd.Series) -> "BaselineModel":
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return data["dark_ratio"] >= 0.01
