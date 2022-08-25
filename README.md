# fcdocs

This is a kedro project to classify documents using various text- and
image-based models.

## Installation

You need a recent version of python (3.8+ should work).
Install all system dependencies as defined in [`django-filingcabinet's default.nix`](https://github.com/okfde/django-filingcabinet/blob/master/default.nix).

Then install the dependencies using

```
pip install -r src/requirements.txt
```

## Download sample data

The input data needs to be placed in `data/01_raw`.
The folders in `data/` follow the [layered data engineering convention](https://kedro.readthedocs.io/en/stable/faq/faq.html#what-is-data-engineering-convention).

A script to download the annotated documents from the
[`fcdocs-annotate`](https://github.com/okfde/fcdocs-annotate) api is provided
in `scripts/download_data.py`. Assuming your server runs on 127.0.0.1:8000, 
you can use the following command

```bash
python scripts/download_data.py --document-endpoint http://127.0.0.1:8000/api/document/ --feature-endpoint http://127.0.0.1:8000/api/feature/
```

This project was developed by the [FragDenStaat](https://fragdenstaat.de)-[team](https://fragdenstaat.de/team)
at [Open Knowledge Foundation Deutschland e.V.](okfn.de).
FragDenStaat provides a simple interface to make and publish
freedom-of-information requests to public bodies.

You can use the following script to download a bunch of attachments from the
[FragDenStaat.de-API](https://fragdenstaat.de/api/) 

```bash
python scripts/download_data_fds.py
```

## How to run the pipeline

> ℹ️ Also see the [Configuration](#configuration) section

The project currently consists of three pipelines:

1. `data_processing` (dp): Cleans the input data and calculates some features from it
2. `classifier` (cf): Trains and evaluates a classification model
3. `clustering` (cs): Trains and evaluates a clustering model

You can run them all using

```
kedro run
```

To run only one of the pipelines you can add the `--pipeline` parameter with the
short-name of a pipeline (`dp`, `cf`, `cs`)
For example to only run the classifier pipeline, use

```
kedro run --pipeline cf
```

> ℹ️ You need to run the data processing pipeline at least once before running
> the model pipelines

## Configuration

You can configure the used models and their parameters in the `conf/base/parameters.yml` file.

By default, the data processing pipeline uses 4 threads for pdf conversion.
If you have more cpu cores, you can change this number by creating a
`conf/local/parameters.yml` file with the following content (replace YOUR_NUMBER_OF_WORKERS):

```yaml
data_processing:
  max_workers: YOUR_NUMBER_OF_WORKERS
```

## Run tests

You can run tests (see `src/tests/`) with

```
kedro test
```

> Note: We currently don't have tests

## Locking project dependencies

To generate or update the dependency requirements run:

```
kedro build-reqs
```

This will `pip-compile` the contents of `src/requirements.txt` into a new file
`src/requirements.lock`. You can see the output of the resolution by opening
`src/requirements.lock`.

[Further information about project dependencies](https://kedro.readthedocs.io/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)


## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r src/requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to convert notebook cells to nodes in a Kedro project
You can move notebook code over into a Kedro project structure using a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#release-5-0-0) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`:

```
kedro jupyter convert <filepath_to_my_notebook>
```
> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. Run the following command to convert all notebook files found in the project root directory and under any of its sub-folders:

```
kedro jupyter convert --all
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.


## Further Documentation

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

## Packaging

[Further information about building project documentation and packaging your project](https://kedro.readthedocs.io/en/stable/tutorial/package_a_project.html)

## Testing the models
### Classification

After you trained a classification with `kedro run --pipeline cf`, you can test it with on some pdfs using

```shell
kedro predict-with-classifier data/06_models/classifier/ YOUR_PDF1 [YOUR_PDF2 ...]
```

This will load the newest version of your model and make a prediction on your pdf files.

If you want to use a specific version of your model, you can specify it using the `--load-version` option:

```shell
kedro predict-with-classifier data/06_models/classifier/ --load-version 2022-07-08T21.22.07.918Z YOUR_PDFS
```

### Clustering

After you trained a clustering with `kedro run --pipeline cs`, you can test it with on some pdfs using

```shell
kedro predict-with-clustering data/06_models/clustering/ YOUR_PDF1 [YOUR_PDF2 ...]
```

This will load the newest version of your model and make a prediction on your pdf files.

If you want to use a specific version of your model, you can specify it using the `--load-version` option:

```shell
kedro predict-with-classifier data/06_models/clustering/ --load-version 2022-07-08T21.22.07.918Z YOUR_PDFS
```
