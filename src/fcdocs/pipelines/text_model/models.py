import json
import logging
import random
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import spacy
import spacy.cli
import spacy.training
import spacy.util


class BaselineModel:
    def fit(self, data, targets) -> "BaselineModel":
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return data["text"].str.lower().str.contains("bescheid")


class SpacyModel:
    def __init__(self, config_path: str, trained_pipeline_name: str):
        self.trained_pipeline_name = trained_pipeline_name
        self.nlp = spacy.load(trained_pipeline_name)
        self.config_path = Path(config_path)
        self.tmpdir = Path(tempfile.mkdtemp())
        self.trained_model = None

    def fit(self, data, targets) -> "SpacyModel":
        logger = logging.getLogger(__name__)

        docs = []
        for (_, row), label in zip(data.iterrows(), targets):
            doc = row.spacy_doc
            if label:
                doc.cats["LABEL"] = True
                doc.cats["NOT_LABEL"] = False
            else:
                doc.cats["LABEL"] = False
                doc.cats["NOT_LABEL"] = True
            docs.append(doc)
        random.seed(0)
        random.shuffle(docs)

        logger.info("Writing corpus for training")
        split = int(len(docs) * 0.9)
        train_bin = spacy.tokens.DocBin(docs=docs[:split])
        train_path = self.tmpdir / "train.spacy"
        train_bin.to_disk(train_path)
        dev_bin = spacy.tokens.DocBin(docs=docs[split:])
        dev_path = self.tmpdir / "dev.spacy"
        dev_bin.to_disk(dev_path)

        overrides = {
            "paths.train": str(train_path),
            "paths.dev": str(dev_path),
        }

        spacy.cli.train.train(
            self.config_path, self.tmpdir / "model", overrides=overrides
        )
        self.trained_model = spacy.load(self.tmpdir / "model" / "model-best")

        shutil.rmtree(self.tmpdir)
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        df = pd.DataFrame(x.cats for x in self.trained_model.pipe(data.text))
        return df.LABEL > df.NOT_LABEL

    def save(self, path: Path):
        with open(path / "args.json", "w") as f:
            json.dump(
                {
                    "trained_pipeline_name": self.trained_pipeline_name,
                    "config_path": str(self.config_path),
                },
                f,
            )
        self.trained_model.to_disk(path / "spacy_model")

    @classmethod
    def load(cls, path: Path):
        with open(path / "args.json") as f:
            args = json.load(f)
        model = cls(args)
        model.trained_model = spacy.load(path / "spacy_model")
