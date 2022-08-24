from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.cluster import KMeans


class BaselineModel:
    """BaselineModel that clusters documents by the first character that appears in them"""

    def fit(self, data) -> "BaselineModel":
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        processed_data = self.process(data)
        pred = processed_data[:, 0]
        return pred

    def save(self, path: Path):
        pass

    @classmethod
    def load(cls, path: Path):
        return cls()

    def process(self, data: pd.DataFrame) -> np.ndarray:
        processed_data = (
            data.text.str[:10]
            .str.strip()
            .str.ljust(10, "a")
            .apply(lambda s: np.array([ord(x) for x in s]))
        )
        return np.vstack(processed_data.values)


class SpacyKMeansModel:
    def __init__(self, n_clusters=8):
        self._model = KMeans(n_clusters=n_clusters)

    def fit(self, data: pd.DataFrame) -> "BaselineModel":
        self._model.fit(self.process(data))
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self._model.predict(self.process(data)))

    def process(self, data: pd.DataFrame) -> np.ndarray:
        vectors = data.spacy_doc.apply(lambda x: x.vector)
        return np.vstack(vectors.values)

    def save(self, path: Path):
        path.mkdir()
        dump(self._model, path / "model.joblib")

    @classmethod
    def load(cls, path: Path):
        model = cls()
        model._model = load(path / "model.joblib")
        return model
