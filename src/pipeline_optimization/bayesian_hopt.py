import numpy as np
import pandas as pd

import plotly.graph_objects as go

from functools import partial
from hyperopt import hp, tpe, fmin,  STATUS_OK, Trials, rand
from hyperopt.fmin import generate_trials_to_calculate


class Config:
    """Space in one dimension"""
    def __init__(self, scope, scale='linear', granularity=None, rounding=None):
        """Initializes a one-dimensional search space"""
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
    """Bayesian hyperparameter optimization"""

    def __init__(self, identifier, search_space, objective, max_evals, algo='tpe', nr_random_starts=20, warmstarter=None):
        """Initializes Bayesian hyperparameter optimization instance."""
        self._identifier = identifier
        self._search_space = search_space
        self.objective = objective
        self.max_evals = max_evals
        if algo == 'tpe':
            self._algo = partial(tpe.suggest, n_startup_jobs=nr_random_starts)
        elif algo == 'random':
            self._algo = rand.suggest
        self._warmstarter = warmstarter
        self.results = None

    @property
    def identifier(self):
        return self._identifier

    def get_numpy_space(self):
        space = self._search_space
        real_space = {}
        for param in space.keys():
            if space[param]._scale == 'linear':
                real_space[param] = np.linspace(space[param].scope[0], space[param].scope[1], space[param].granularity)
            if space[param]._scale == 'log':
                real_space[param] = np.logspace(space[param].scope[0], space[param].scope[1], space[param].granularity)
            real_space[param] = np.round(real_space[param], space[param]._rounding)
        return real_space

    def hyperopt_objective(self, unit_params):

        # get real space
        real_space = self.get_numpy_space()
        real_params = {key: real_space[key][int(unit_params[key]-1)] for key in self._search_space.keys()}

        # perform evaluation
        result, walltime, crossval = self.objective(real_params)

        return {'loss': result, 'status': STATUS_OK, 'walltime': walltime, 'crossval': crossval, 'params': real_params}

    def run_bayesian_hopt(self, time_series=None, show_progressbar=True):
        """Runs the Bayesian hyperparameter optimization."""

        time_series = pd.DataFrame(time_series)

        if time_series.empty and self._warmstarter:
            raise ValueError('A warmstart requires an input time series to derive a suggestion from.')

        # Create trials object to store information on optimization process
        if self._warmstarter:
            warmstart_configs = self._warmstarter.suggest(time_series)
            real_space = self.get_numpy_space()
            unit_params = [{key: np.abs(real_space[key]-config[key]).argmin()+1 for key in real_space.keys()} for config in warmstart_configs]
            trials = generate_trials_to_calculate(unit_params)
            hyperopt_evals = self.max_evals - len(warmstart_configs)
        else:
            trials = Trials()
            hyperopt_evals = self.max_evals

        # Create the hyperopt format arguments
        space = self._search_space
        hyperopt_space = {key: hp.quniform(key, 1, space[key].granularity, 1) for key in list(space.keys())}

        # Run the hyperopt optimization
        fmin(
            fn=self.hyperopt_objective,
            space=hyperopt_space,
            algo=self._algo,
            max_evals=hyperopt_evals,
            trials=trials,
            show_progressbar=show_progressbar
        )

        results = pd.DataFrame()
        for trial in trials.trials:
            result = trial['result']
            params = result.pop('params')
            result = pd.concat([pd.Series(result), pd.Series(params)], keys=['results', 'configs'])
            results = results.append(result, ignore_index=True)
        results.columns = pd.MultiIndex.from_tuples(results.columns)

        # add results as attribute to the instance
        self.results = results

        return results

    def visualize_search_performance(self, xaxis='iterations', all_losses=False, crossvalidation=False):
        results = self.results

        # create figure
        fig = go.Figure()

        # define x-axis
        if xaxis == 'iterations':
            idx = results.index
        if xaxis == 'walltime':
            idx = pd.Series([results.loc[:i, ('results', 'walltime')].sum() for i in range(len(results))])

        # compute best-so-far series
        rolling_min = pd.Series([results.loc[:i, ('results', 'loss')].min() for i in range(len(results))])
        fig.add_trace(go.Scatter(x=idx, y=rolling_min, mode='lines', name='Best so far'))

        # optionally show all computed evaluations
        if all_losses:
            loss = results.loc[:, ('results', 'loss')]
            fig.add_trace(go.Scatter(x=idx, y=loss, mode='markers', name='Iteration result'))

        # optionally show the crossvalidation performance
        if crossvalidation:
            rolling_crossval = pd.Series([results.loc[:i, ('results', 'crossval')].min() for i in range(len(results))])
            fig.add_trace(go.Scatter(x=idx, y=rolling_crossval, mode='lines', name='Crossvalidation so far'))

        # optionally show all computed evaluations of crossvalidation performance
        if all_losses & crossvalidation:
            crossval = results.loc[:, ('results', 'crossval')]
            fig.add_trace(go.Scatter(x=idx, y=crossval, mode='markers', name='Crossvalidation result'))

        fig.show()
