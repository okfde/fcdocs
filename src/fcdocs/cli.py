"""Command line tools for manipulating a Kedro project.
Intended to be invoked via `kedro`."""
import shutil
from pathlib import Path
from typing import List, Optional

import click
from kedro.config import ConfigLoader
from kedro.framework.cli.project import project_group
from kedro.framework.cli.utils import CONTEXT_SETTINGS
from kedro.framework.project import settings
from kedro.io import DataCatalog, MemoryDataSet, Version
from kedro.runner import SequentialRunner
from rich.console import Console
from rich.table import Table

from fcdocs.extras.datasets.document_dataset import DocumentDataSet
from fcdocs.pipelines import data_processing as dp
from fcdocs.pipelines.classifier.model_dataset import ModelDataSet
from fcdocs.pipelines.clustering.model_dataset import (
    ModelDataSet as ClusteringModelDataSet,
)


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


def get_document_df(pdf_files: List[Path]):
    conf_path = str(settings.CONF_SOURCE)
    conf_loader = ConfigLoader(conf_source=conf_path, env="local")
    conf_parameters = conf_loader.get("parameters*")

    dp_config = conf_parameters["data_processing"]

    data_processing_pipeline = dp.create_pipeline()
    pre_pipeline = data_processing_pipeline.from_inputs("pdfdocuments").to_outputs(
        "data_with_features"
    )
    io = DataCatalog(
        {
            "pdfdocuments": MemoryDataSet(),
            "data_with_features": MemoryDataSet(),
            "params:data_processing.max_workers": MemoryDataSet(),
            "params:data_processing.spacy_model": MemoryDataSet(),
        }
    )
    io.save(
        "pdfdocuments",
        {str(pdf_file): DocumentDataSet(pdf_file).load for pdf_file in pdf_files},
    )
    io.save("params:data_processing.max_workers", dp_config["max_workers"])
    io.save("params:data_processing.spacy_model", dp_config["spacy_model"])

    runner = SequentialRunner()
    runner.run(pipeline=pre_pipeline, catalog=io)

    return io.datasets.data_with_features.load()


@project_group.command()
@click.argument("model", type=Path)
@click.option("--load-version", default=None)
@click.argument("pdf_files", type=Path, nargs=-1, required=True)
def predict_with_classifier(model, load_version, pdf_files):
    version = Version(load_version, None)
    model = ModelDataSet(model, version).load()

    prediction, score = model.predict(get_document_df(pdf_files))

    table = Table(title="Class Predictions")
    table.add_column("Filename", justify="left")
    table.add_column("Prediction", justify="right")
    table.add_column("Score", justify="right")

    for file, pred, sc in zip(pdf_files, prediction, score):
        table.add_row(str(file), "Yes" if pred else "No", "{0:.2%}".format(sc))

    console = Console()
    console.print(table)


@project_group.command()
@click.argument("model", type=Path)
@click.option("--load-version", default=None)
@click.argument("pdf_files", type=Path, nargs=-1, required=True)
def predict_with_clustering(model, load_version, pdf_files):
    version = Version(load_version, None)
    model = ClusteringModelDataSet(model, version).load()

    prediction = model.predict(get_document_df(pdf_files))

    table = Table(title="Cluster Predictions")
    table.add_column("Filename", justify="left")
    table.add_column("Prediction", justify="right")

    for file, pred in zip(pdf_files, prediction):
        table.add_row(str(file), str(pred))

    console = Console()
    console.print(table)


@project_group.command()
@click.argument("model", type=Path)
@click.option("--load-version", default=None)
@click.argument("out_dir", type=Path)
@click.argument("pdf_files", type=Path, nargs=-1, required=True)
def predict_clusters_into_folder(
    model: Path, load_version: Optional[str], out_dir: Path, pdf_files: List[Path]
):
    version = Version(load_version, None)
    model = ClusteringModelDataSet(model, version).load()

    prediction = model.predict(get_document_df(pdf_files))

    out_dir.mkdir(exist_ok=True)
    for file, pred in zip(pdf_files, prediction):
        cluster_dir = out_dir / str(pred)
        cluster_dir.mkdir(exist_ok=True)
        out_file = cluster_dir / file.name
        shutil.copy(file, out_file)


@project_group.command()
@click.argument("model", type=Path)
@click.option("--load-version", default=None)
@click.argument("out_file", type=Path)
def package_model(model: Path, load_version: Optional[str], out_file: Path):
    version = Version(load_version, None)
    model = ClusteringModelDataSet(model, version)

    model.zip_into(out_file)
