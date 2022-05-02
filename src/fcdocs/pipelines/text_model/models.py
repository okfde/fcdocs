import pandas as pd


class BaselineModel:
    def fit(self, data, targets):
        pass

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return data["text"].str.lower().str.contains("bescheid")
