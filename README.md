Warmstart
==============================

This python package performs warmstart experiments in the AutoML research domain. It performs experiments comparing search strategies such as warmstarted Bayesian optimization, following the formalization of \figref{label:formalization}. We provide a quick-start user example of the experiment of this specific research.


Quickstart
------------
### Environment setup
Start by cloning the Github repository (https://github.com/JeroenSwart/autoxgb) and installing Jupyter Lab, an extensible environment for interactive and reproducible computing. The README file gives a walk-through for setting up the virtual environment and creating an ipython kernel, from which to launch a Jupyter Lab notebook. Firstly the classes are imported from the library.

```
# Import external libraries
import pickle

# Import internal metalearning libraries
from src.experimenting.hopt_experiment import HoptExperiment
from src.metalearning.metadata import MetaDataset
from src.metalearning.warmstarter import Warmstarter
from src.pipeline_optimization.bayesian_hopt import BayesianHopt
from src.utils.metafeature_utils import size, cumac

# Import thesis specific objective and search space
from src.utils.thesis_utils import thesis_lookup_objective, thesis_search_space
```

### Fixed experiment setting
We then define the fixed setting of the experiment. A metasample is instantiated with an identifier string, the training dataset, the test dataset and results (a pandas dataframe of pipeline configurations and resulting performance indices). In this example pickle loads prestored instances of metasamples. A metadataset is instantiated with a list of metasamples and a list of metafeature functions, mappings from a dataset to a metafeature. Futhermore the identifier of target datasets, the search space and the objective, which is the mapping from pipeline configuration to the performance index, are defined. The maximum number of iterations for the pipeline optimizations is specified in max\_evals, the number of duplicates is defined and the size of the initialization batch is defined in n_init_configs.
\newline

```
# Initialize metadataset and calculate metafeatures
metadataset_sample_names = !ls ../../data/metadata/interim
metasamples = [pickle.load(open('../../data/metadata/interim/' + sample_name,"rb")) for sample_name in metadataset_sample_names]
metadataset = MetaDataset(metasamples, metafeature_functions=[size, cumac])
objective = thesis_lookup_objective
search_space = thesis_search_space()
target_ids = [metasample.identifier for metasample in metadataset.metasamples]

# Experiment practicalities
max_evals = 50
duplicates = 2
n_init_configs = 5
```

### Variable experiment setting}
The variable experiment setting defines the compared search strategies. Since the current implementation integrates with the open-source library hyperopt, search strategies are limited to Bayesian optimization (BayesianHopt), which is instantiated with an identifier, search space, objective and a maximum number of iterations. A naive Bayesian hyperoptimization has an initial set of random configurations, defined by nr\_random\_starts. The search strategy is a random search if nr\_random\_starts is set equal to max\_evals.
\newline

```
# initialize search strategies
rand = BayesianHopt(
    identifier='Random search',
    search_space=search_space,
    objective=objective,
    max_evals=max_evals,
    nr_random_starts=max_evals
)
naive = BayesianHopt(
    identifier='Naive Bayesian optimization',
    search_space=search_space,
    objective=objective,
    max_evals=max_evals,
    nr_random_starts=n_init_configs
)
```

The Bayesian hyperoptimization can be given a warmstarter object, instantiated with the defined metadataset, the number of most similar samples and the number of best configurations per sample and the number of suggested initialization configurations.
\newline

```
warm = BayesianHopt(
    identifier='Warmstarted Bayesian optimization',
    search_space=search_space,
    objective=objective,
    max_evals=max_evals,
    warmstarter=Warmstarter(metadataset, n_init_configs=n_init_configs, n_sim_samples=5, n_best_per_sample=5)
)
cold = BayesianHopt(
    identifier='Coldstarted Bayesian optimization',
    search_space=search_space,
    objective=objective,
    max_evals=max_evals,
    warmstarter=Warmstarter(metadataset, n_init_configs=n_init_configs, n_sim_samples=5, n_best_per_sample=5, cold=True),
)
```

The experiment is then instantiated by giving it the Bayesian hyperoptimizations and the number of duplicates to average over.
\newline

```
# initialize hyperoptimization experiment
hopt_exp = HoptExperiment(
    hopts=[rand, naive, warm, cold],
    duplicates=duplicates,
    objective=objective,
    metadataset=metadataset
)
```

### Results
Experiment results in a dataframe are added as an attribute by calling the run function on the pipeline optimization experiment.
\newline

```
hopt_exp.run_hopt_experiment(target_ids)
```

Several visualizers show the experiment results, for example the averaged performance over duplicates so far, with respect to the amount of iterations of the search strategy, as shown in \figref{fig:vis_ex}.
\newline

```
hopt_exp.visualize_avg_performance(target_ids[12])
```

### Instructions for making a notebook kernel

Install the ipython kernel so we have exactly the same packages, versions and extensions!

```
cd autoxgb
export VENV_PATH="../autoxgb_venv"
virtualenv -p python3 $VENV_PATH
source ../autoxgb_venv/bin/activate
pip3 install -r requirements.txt
pre-commit install
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
pre-commit install
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