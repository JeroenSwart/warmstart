import pandas as pd
from tqdm import tqdm


class MetaSample:
    def __init__(self, identifier, train_dataset, test_dataset, results=None):
        self._identifier = identifier
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._results = results

    @property
    def identifier(self):
        return self._identifier

    @property
    def time_series(self):
        return self._train_dataset

    @property
    def test_time_series(self):
        return self._test_dataset

    @property
    def results(self):
        return self._results

    def metafeatures(self, metafeature_functions):
        metafeatures = pd.Series(
            data=[calc(self.time_series) for calc in metafeature_functions],
            index=[calc.__name__ for calc in metafeature_functions],
        )
        return metafeatures

    def get_best_hyperparameters(self, nr_best):
        best_configs_df = self.results.sort_values(by=[("diagnostics", "mae")]).iloc[
            :nr_best
        ]["hyperparameters"]
        best_configs = [best_configs_df.iloc[i].to_dict() for i in range(nr_best)]
        return best_configs

    def get_best_performance(self):
        best_perf = self.results.sort_values(by=[("diagnostics", "mae")]).iloc[0][
            ("diagnostics", "mae")
        ]
        return best_perf


class MetaDataset:
    def __init__(self, metasamples, metafeature_functions):
        self.metasamples = metasamples
        self.metafeature_functions = metafeature_functions
        self.metafeature_set = pd.DataFrame(
            data=[
                metasample.metafeatures(metafeature_functions)
                for metasample in tqdm(
                    self.metasamples, desc="Calculate metafeatures of metasamples"
                )
            ],
            index=[metasample.identifier for metasample in self.metasamples],
        )
