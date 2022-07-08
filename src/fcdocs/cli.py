"""Command line tools for manipulating a Kedro project.
Intended to be invoked via `kedro`."""
from pathlib import Path

import click
import pandas as pd
from kedro.framework.cli.project import project_group
from kedro.framework.cli.utils import CONTEXT_SETTINGS
from kedro.io import Version

from fcdocs.extras.datasets.document_dataset import DocumentDataSet
from fcdocs.pipelines.text_model.model_dataset import ModelDataSet


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


@project_group.command()
@click.argument("model", type=Path)
@click.option("--load-version", default=None)
@click.argument("pdf_file", type=Path)
def predict_with_text_model(model, load_version, pdf_file):
    document = DocumentDataSet(pdf_file).load()
    version = Version(load_version, None)
    model = ModelDataSet(model, version).load()

    prediction = model.predict(pd.DataFrame([document]))[0]
    print("Prediction:", "Yes" if prediction else "No")
