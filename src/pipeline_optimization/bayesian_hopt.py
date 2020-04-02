import numpy as np
import pandas as pd

from functools import partial
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials, rand
from hyperopt.fmin import generate_trials_to_calculate


class Config:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, scope, scale="linear", granularity=None, rounding=None):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
        self._scope = scope
        self._scale = scale
        self._granularity = granularity
        self._rounding = rounding

    @property
    def scope(self):
        return self._scope

    @property
    def granularity(self):
        return self._granularity


class BayesianHopt:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

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
        """Initializes Bayesian hyperparameter optimization instance.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

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

    def get_numpy_space(self):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        space = self._search_space
        real_space = {}
        for param in space.keys():
            if space[param]._scale == "linear":
                real_space[param] = np.linspace(
                    space[param].scope[0],
                    space[param].scope[1],
                    space[param].granularity,
                )
            if space[param]._scale == "log":
                real_space[param] = np.logspace(
                    space[param].scope[0],
                    space[param].scope[1],
                    space[param].granularity,
                )
            real_space[param] = np.round(real_space[param], space[param]._rounding)
        return real_space

    def hyperopt_objective(self, unit_params):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

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

    def run_bayesian_hopt(self, time_series=None, show_progressbar=True):
        """Runs the Bayesian hyperparameter optimization.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        time_series = pd.DataFrame(time_series)

        if time_series.empty and self._warmstarter:
            raise ValueError(
                "A warmstart requires an input time series to derive a suggestion from."
            )

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

        # Run the hyperopt optimization
        fmin(
            fn=self.hyperopt_objective,
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
