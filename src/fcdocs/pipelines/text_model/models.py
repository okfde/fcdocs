import pandas as pd


class BaselineModel:
    def fit(self, data, targets) -> "BaselineModel":
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return data["text"].str.lower().str.contains("bescheid")
