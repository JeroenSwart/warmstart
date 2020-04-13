import numpy as np
import pandas as pd

from functools import partial
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials, rand
from hyperopt.fmin import generate_trials_to_calculate


class Config:
    # todo: is configuration the right naming? I mean it as 'one of the set of configurations'. Maybe back to hyperparameters?
    # todo: change config to superpameter after feedback Peyman.
    """A Config is the definition of the search space in one dimension.

    This base abstraction is used for compatibility with the variable definitions of search spaces in external packages
    for pipeline optimization.
    """

    def __init__(self, start, stop, scale="lin", granularity=None, rounding=None):
        """A Config is instantiated with a starting value, stopping value and a scale of the space. The granularity and
        a rounding function are optional arguments.

        Args:
            start (float): the starting value of the sequence.
            stop (float): the end value of the sequence.
            scale ('lin' or 'log', optional): the prior on this dimension.
            granularity (int, optional): the number of evenly spaced samples over the interval [`start`, `stop`].
            rounding (int, optional): the given number of decimals to round the array to.

        """
        self._start = start
        self._stop = stop
        self._scale = scale
        self._granularity = granularity
        self._rounding = rounding

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def scale(self):
        return self._scale

    @property
    def granularity(self):
        return self._granularity

    @property
    def rounding(self):
        return self._rounding


class BayesianHopt:
    # todo: change name because we do pipeline optimizations in stead of hyperparameter optimizations
    """A BayesianHopt (Bayesian hyperparameter optimization) is an optimization algorithm for pipeline configurations.

    Bayesian optimization is widely used for optimizing computationally expensive black box functions. It attempts to
    reduce the number of expensive evaluations, by making intelligent suggestions in the search space. The algorithm
    has an urge to exploit regions with the best evaluations scores, and to explore the regions where few evaluations
    have been performed.

    The intelligent suggestions come from a model trained on the historical evaluations of the objective function. The
    proposed suggestions from the model are evaluated by an acquisition function, to choose the most promising
    suggestion.

    This implementation supports the hyperopt package, which uses the Tree Parzen Estimators as model with the
    expected improvement as acquisition function.

    Attribute:
        results(pd.DataFrame): pipeline configuration and evaluation info from search iterations.

    """

    def __init__(
        self,
        identifier,
        search_space,
        objective,
        max_evals,
        algo="tpe",
        nr_random_starts=20,
        warmstarter=None,
    ):
        """A BayesianHopt is instantiated with a unique identifier, a search space, an objective function and a maximum
        number of search iterations. Optionally, the search strategy can be switched to random search, the number of
        initial random starts can be specified and the search strategy can be boosted with a warmstarter.

        Args:
            identifier (str): a unique identifier for this BayesianHopt instance.
            search_space (dict): the space of pipeline configurations to search.
            objective (function): the mapping from a pipeline configuration to performance index.
            max_evals (int): the number of search iterations after which to stop.
            algo ('tpe' or 'random'): choice between Tree Parzen Estimator or random search.
            nr_random_starts (int): the number of random starts to begin with.
            # todo: what are the docstring rules for referring to another object, s.t. an 'unexpected type' warning
                pops up when the input type is wrong?
            warmstarter (Warmstarter): the metalearning model that suggests initial pipeline configurations.

        Search_space example:

            search_space = {
                "num_trees": Config(100, 800, granularity=6, rounding=1),
                "learning_rate": Config(-2.5, -0.5, granularity=10, scale="log", rounding=13),
                "max_depth": Config(5, 20, granularity=8, rounding=0),
                "min_child_weight": Config(5, 40, granularity=3, rounding=1),
                "subsample": Config(0.5, 1.0, granularity=3, rounding=2),
            }

        """
        self._identifier = identifier
        self._search_space = search_space
        self.objective = objective
        self.max_evals = max_evals
        if algo == "tpe":
            self._algo = partial(tpe.suggest, n_startup_jobs=nr_random_starts)
        elif algo == "random":
            self._algo = rand.suggest
        self._warmstarter = warmstarter
        self.results = None

    # todo: introduce possibility to not input an objective during initialization, but check for an objective when the
    #   bayesian hyperparameter optimization is run

    @property
    def identifier(self):
        return self._identifier

    # todo: seems like this little guy should be in a search space class, which does not yet exist..?
    def get_numpy_space(self):
        """Returns the search space in the following format:

        search_space = {
            'num_trees': array([100., 240., 380., 520., 660., 800.]),
            'learning_rate': array([0.00316228, 0.005275  , 0.00879923, 0.01467799, 0.02448437,0.04084239, 0.06812921]),
            'max_depth': array([ 5.,  7.,  9., 11., 14., 16., 18., 20.]),
            'min_child_weight': array([ 5. , 22.5, 40. ]),
            'subsample': array([0.5 , 0.75, 1.  ])
        }

        """
        real_space = {}
        for param in self._search_space.keys():
            config = self._search_space[param]
            if config.scale == "lin":
                real_space[param] = np.linspace(
                    config.start, config.stop, config.granularity
                )
            if config.scale == "log":
                real_space[param] = np.logspace(
                    config.start, config.stop, config.granularity
                )
            real_space[param] = np.round(real_space[param], config.rounding)
        return real_space

    def _hyperopt_objective(self, unit_params):
        """This method overcomes search space compatibility issues of the hyperopt package, which is not able to
        discretize a search space to non-integer values. Hyperopt is given an integer search space, and this search
        space is transformed back into the real (non-integer) search space format in this objective function.

        """
        # get real space
        real_space = self.get_numpy_space()
        real_params = {
            key: real_space[key][int(unit_params[key] - 1)]
            for key in self._search_space.keys()
        }

        # perform evaluation
        result, walltime, crossval = self.objective(real_params)

        return {
            "loss": result,
            "status": STATUS_OK,
            "walltime": walltime,
            "crossval": crossval,
            "params": real_params,
        }

    def run_bayesian_hopt(self, time_series, show_progressbar=True):
        """Runs the Bayesian hyperparameter optimization. The results are stored as an attribute.

        Args:
            time_series (pd.DataFrame): the dataset for which to optimize the superparameters
            show_progressbar (boolean, optional): show bar for the progress of the superoptimization

        """
        time_series = pd.DataFrame(time_series)

        if time_series.empty and self._warmstarter:
            raise ValueError(
                "A warmstart requires an input time series to derive a suggestion from."
            )

        # todo: assert that time_series has the right format for the objective function.

        # Create trials object to store information on optimization process
        if self._warmstarter:
            warmstart_configs = self._warmstarter.suggest(time_series)
            real_space = self.get_numpy_space()
            unit_params = [
                {
                    key: np.abs(real_space[key] - config[key]).argmin() + 1
                    for key in real_space.keys()
                }
                for config in warmstart_configs
            ]
            trials = generate_trials_to_calculate(unit_params)
            hyperopt_evals = self.max_evals - len(warmstart_configs)
        else:
            trials = Trials()
            hyperopt_evals = self.max_evals

        # Create the hyperopt format arguments
        space = self._search_space
        hyperopt_space = {
            key: hp.quniform(key, 1, space[key].granularity, 1)
            for key in list(space.keys())
        }

        # Run the hyperopt optimization, note that is in unit format, due to limitations of search spaces in hyperopt.
        # The search space is transformed back in _hyperopt_objective.
        fmin(
            fn=self._hyperopt_objective,
            space=hyperopt_space,
            algo=self._algo,
            max_evals=hyperopt_evals,
            trials=trials,
            show_progressbar=show_progressbar,
        )

        results = pd.DataFrame()
        for trial in trials.trials:
            result = trial["result"]
            params = result.pop("params")
            result = pd.concat(
                [pd.Series(result), pd.Series(params)], keys=["results", "configs"]
            )
            results = results.append(result, ignore_index=True)
        results.columns = pd.MultiIndex.from_tuples(results.columns)

        # add results as attribute to the instance
        self.results = results

        return results
