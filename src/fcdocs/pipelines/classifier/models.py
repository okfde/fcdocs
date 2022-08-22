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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from joblib import dump, load


class BaselineModel:
    def fit(self, data, targets) -> "BaselineModel":
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return data["text"].str.lower().str.contains("bescheid")

    def save(self, path: Path):
        pass

    @classmethod
    def load(cls, path: Path):
        return cls()


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
        path.mkdir()
        with open(path / "kwargs.json", "w") as f:
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
        with open(path / "kwargs.json") as f:
            kwargs = json.load(f)
        model = cls(**kwargs)
        model.trained_model = spacy.load(str(path / "spacy_model"))
        return model


class RandomForestClassifierModel:
    def fit(self, data, targets) -> "RandomForestClassifierModel":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidfconverter = TfidfVectorizer(
            max_features=1500,
            min_df=5,
            max_df=0.7,
            # stop_words=stopwords.words("english"),
        )
        X = self._tfidfconverter.fit_transform(data.text).toarray()

        self._classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
        self._classifier.fit(X, targets)
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        X = self._tfidfconverter.transform(data.text).toarray()
        return pd.Series(self._classifier.predict(X))

    def save(self, path: Path):
        path.mkdir()
        dump(self._classifier, path / "classifier.joblib")
        dump(self._tfidfconverter, path / "tfidfconverter.joblib")

    @classmethod
    def load(cls, path: Path):
        model = cls()
        model._classifier = load(path / "classifier.joblib")
        model._tfidfconverter = load(path / "tfidfconverter.joblib")
        return model


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        width = 6
        num_classes = 2
        self.conv1 = nn.Conv2d(1, width, 3, 1)
        self.conv2 = nn.Conv2d(width, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(33856, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.adaptive_max_pool2d(x, (23, 23))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_device():
    if torch.cuda.is_available():
        dev_name = "cuda:0"
    elif torch.backends.mps.is_available():
        dev_name = "mps"
    else:
        dev_name = "cpu"

    device = torch.device(dev_name)
    return device


class ImageSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, image_series, targets, device, trans):
        print("myDataset init")
        self.x = (
            image_series.apply(lambda x: x.convert("L"))
            .apply(trans)
            .apply(lambda x: x.to(device))
        )
        self.y = targets.astype("int").apply(torch.tensor).apply(lambda x: x.to(device))
        self.x.index = pd.RangeIndex(len(self.x.index))
        self.y.index = pd.RangeIndex(len(self.y.index))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train(optimizer, net, train_loader):
    net.train()
    for (data, target) in train_loader:
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


class CNNImageModel:
    def __init__(self):
        self.device = get_device()
        self.net = Net().to(self.device)
        self.trans = transforms.ToTensor()

    def fit(self, data, targets) -> "CNNImageModel":
        # Training settings
        n_epochs = 1
        batch_size_train = 1
        learning_rate = 0.01
        momentum = 0.5

        random_seed = 1
        torch.manual_seed(random_seed)

        loader = torch.utils.data.DataLoader(
            ImageSeriesDataset(
                image_series=data.image,
                targets=targets,
                device=self.device,
                trans=self.trans,
            ),
            batch_size=batch_size_train,
            shuffle=True,
        )

        self.optimizer = optim.SGD(
            self.net.parameters(), lr=learning_rate, momentum=momentum
        )

        for _ in range(n_epochs):
            train(self.optimizer, self.net, loader)

        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        bw_imgs = data.image.apply(lambda x: x.convert("L"))
        image_trans = bw_imgs.apply(self.trans)

        outputs = []
        with torch.no_grad():
            for image in image_trans.values:
                image_tensor = torch.stack([image])
                image_tensor = image_tensor.to(self.device)
                outputs.append(self.net(image_tensor))

        outputs = torch.cat(outputs)
        predictions = outputs.argmax(dim=1)
        return pd.Series(predictions.cpu())

    def save(self, path: Path):
        torch.save(self.net.state_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "CNNImageModel":
        model = cls()
        model.net.load_state_dict(torch.load(str(path), map_location=get_device()))
        return model


class BaselineImageModel:
    def fit(self, data: pd.DataFrame, targets: pd.Series) -> "BaselineImageModel":
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return data["dark_ratio"] >= 0.01

    def save(self, path: Path):
        pass

    @classmethod
    def load(cls, path: Path):
        return cls()
