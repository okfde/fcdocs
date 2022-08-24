"""Command line tools for manipulating a Kedro project.
Intended to be invoked via `kedro`."""
from pathlib import Path

import click
from kedro.config import ConfigLoader
from kedro.framework.cli.project import project_group
from kedro.framework.cli.utils import CONTEXT_SETTINGS
from kedro.framework.project import settings
from kedro.io import DataCatalog, MemoryDataSet, Version
from kedro.runner import SequentialRunner

from fcdocs.extras.datasets.document_dataset import DocumentDataSet
from fcdocs.pipelines import data_processing as dp
from fcdocs.pipelines.classifier.model_dataset import ModelDataSet
from fcdocs.pipelines.clustering.model_dataset import (
    ModelDataSet as ClusteringModelDataSet,
)


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


def get_document_df(pdf_file: Path):

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
    io.save("pdfdocuments", {"1": DocumentDataSet(pdf_file).load})
    io.save("params:data_processing.max_workers", dp_config["max_workers"])
    io.save("params:data_processing.spacy_model", dp_config["spacy_model"])

    runner = SequentialRunner()
    runner.run(pipeline=pre_pipeline, catalog=io)

    return io.datasets.data_with_features.load()


@project_group.command()
@click.argument("model", type=Path)
@click.option("--load-version", default=None)
@click.argument("pdf_file", type=Path)
def predict_with_classifier(model, load_version, pdf_file):
    version = Version(load_version, None)
    model = ModelDataSet(model, version).load()

    prediction = model.predict(get_document_df(pdf_file))[0]
    print("Prediction:", "Yes" if prediction else "No")


@project_group.command()
@click.argument("model", type=Path)
@click.option("--load-version", default=None)
@click.argument("pdf_file", type=Path)
def predict_with_clustering(model, load_version, pdf_file):
    version = Version(load_version, None)
    model = ClusteringModelDataSet(model, version).load()

    prediction = model.predict(get_document_df(pdf_file))[0]
    print("Cluster:", prediction)
