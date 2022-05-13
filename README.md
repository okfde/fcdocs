# fcdocs

This is a kedro project to classify documents using various text- and
image-based models.

## Installation

You need a recent version of python (3.8+ should work).
You also need the `pdftoppm` tool from the poppler utils and `imagemagick`.
Then install the dependencies using

```
pip install -r src/requirements.txt
```

## Download sample data

The input data needs to be placed in `data/01_raw`.
The folders in `data/` follow the [layered data engineering convention](https://kedro.readthedocs.io/en/stable/faq/faq.html#what-is-data-engineering-convention).
You can use the following script to download a bunch of documents from the
[FragDenStaat.de-API](https://fragdenstaat.de/api/) 

```bash
python scripts/download_data.py
```

## How to run the pipeline

The project currently consists of three pipelines:

1. `data_processing` (dp): Cleans the input data and calculates some features from it
2. `image_model` (im): Trains and evaluates an image-based model
3. `text_model` (tm): Trains and evaluates a text-based model

You can run them all using

```
kedro run
```

To run only one of the pipelines you can add the `--pipeline` parameter with the
short-name of a pipeline (`dp`, `im`, `tm`)
For example to only run the image model pipeline, use

```
kedro run --pipeline im
```

> ℹ️ You need to run the data processing pipeline at least once before running
> the model pipelines

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
