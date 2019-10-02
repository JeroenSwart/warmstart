from src.utils.metafeature_utils import size, endogenous_mean, maxminvar, adf

import pandas as pd


class MetaSample:

    def __init__(self, identifier, time_series, results):
        self._identifier = identifier
        self._time_series = time_series
        self._results = results
        self._metafeatures = None

    @property
    def identifier(self):
        return self._identifier

    @property
    def time_series(self):
        return self._time_series

    @property
    def results(self):
        return self._results

    @property
    def metafeatures(self):
        if not self._metafeatures:
            metafeature_functions = [size, endogenous_mean, maxminvar, adf]
            self._metafeatures = pd.Series(
                data=[calc(self.time_series) for calc in metafeature_functions],
                index=[calc.__name__ for calc in metafeature_functions]
            )
        return self._metafeatures

    def get_best_hyperparameters(self, nr_best):
        best_configs_df = self.results.sort_values(by=[('diagnostics', 'mae')]).iloc[:nr_best]['hyperparameters']
        best_configs = [best_configs_df.iloc[i].to_dict() for i in range(nr_best)]
        return best_configs


class MetaDataset:

    def __init__(self, metasamples):
        self.metasamples = metasamples
