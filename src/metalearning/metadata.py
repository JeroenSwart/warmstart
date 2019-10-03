from src.utils.metafeature_utils import size, endogenous_mean, maxminvar, adf, cumac

import pandas as pd
from tqdm import tqdm


class MetaSample:

    def __init__(self, identifier, time_series, results=None):
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
        if self._metafeatures is None:
            metafeature_functions = [size, endogenous_mean, maxminvar, adf, cumac]
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
        self.metafeature_set = pd.DataFrame(
            data=[metasample.metafeatures for metasample in tqdm(self.metasamples, desc='Calculate metafeatures of metasamples')],
            index=[metasample.identifier for metasample in self.metasamples]
        )

    # @property
    # def metafeature_set(self):
    #     if self._metafeature_set is None:
    #         self._metafeature_set = pd.DataFrame(
    #             data=[metasample.metafeatures for metasample in tqdm(self.metasamples, desc='Calculate metafeatures of metasamples')],
    #             index=[metasample.identifier for metasample in self.metasamples]
    #         )
    #     return self._metafeature_set
