import json
from pathlib import Path, PurePosixPath

from kedro.io import AbstractVersionedDataSet

from . import models


class ModelDataSet(AbstractVersionedDataSet):
    def __init__(self, path, version):
        super().__init__(PurePosixPath(path), version)

    def _load(self):
        load_path = self._get_load_path()
        with open(load_path / "meta.json") as f:
            meta = json.load(f)
        model_name = meta["model_name"]
        model = getattr(models, model_name)
        return model.load(load_path / "model")

    def _save(self, model) -> None:
        model_name = model.__class__.__name__
        save_path = Path(self._get_save_path())
        save_path.mkdir(parents=True, exist_ok=False)
        with open(save_path / "meta.json", "w") as f:
            json.dump({"model_name": model_name}, f)
        model.save(save_path / "model")

    def _exists(self) -> bool:
        path = self._get_load_path()
        return Path(path.as_posix()).exists()

    def _describe(self):
        return dict(version=self._version)
