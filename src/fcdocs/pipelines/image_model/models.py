import pandas as pd


class BaselineModel:
    def fit(self, data: pd.DataFrame, targets: pd.Series):
        pass

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return data["dark_ratio"] >= 0.01
