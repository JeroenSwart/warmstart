AutoML
==============================

AutoML for tree-based forecasting

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

### Instructions for making a notebook kernel

Install the ipython kernel so we have exactly the same packages, versions and extensions!

```
cd autoxgb
export VENV_PATH="../autoxgb_venv"
virtualenv -p python3 $VENV_PATH
source ../autoxgb_venv/bin/activate
pip3 install -r requirements.txt
export KERNEL_NAME="autoxgb_kernel"
export DISPLAY_NAME="AutoXGB Notebook"
pip3 install ipykernel
python3 -m ipykernel install --name $KERNEL_NAME --display-name "$DISPLAY_NAME" --user
jupyter labextension install @jupyterlab/toc@0.6.0 --no-build
jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.38.1 --no-build
jupyter labextension install plotlywidget@0.11.0 --no-build
jupyter labextension install @jupyterlab/plotly-extension@1.0.0 --no-build
jupyter labextension install jupyterlab-chart-editor@1.2.0 --no-build
jupyter lab build
deactivate
```

For installing additional packages and storing it in the requirements, run:

```
cd autoxgb
export VENV_yPATH="../autoxgb_venv"
virtualenv -p python3 $VENV_PATH
source ../autoxgb_venv/bin/activate
pip3 install -r requirements.txt
<INSTALL COMMAND>
pip install -e .
pip freeze > requirements.txt
export KERNEL_NAME="autoxgb_kernel"
export DISPLAY_NAME="AutoXGB Notebook"
pip3 install ipykernel
python3 -m ipykernel install --name $KERNEL_NAME --display-name "$DISPLAY_NAME" --user
jupyter labextension install @jupyterlab/toc@0.6.0 --no-build
jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.38.1 --no-build
jupyter labextension install plotlywidget@0.11.0 --no-build
jupyter labextension install @jupyterlab/plotly-extension@1.0.0 --no-build
jupyter labextension install jupyterlab-chart-editor@1.2.0 --no-build
jupyter lab build
deactivate
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
