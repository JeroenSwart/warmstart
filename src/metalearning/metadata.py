import pandas as pd
from tqdm import tqdm


class MetaSample:
    """A MetaSample is a previous task, stored as a dataset and the results of its pipeline optimization.

    MetaSamples are bundled in a MetaDataset to infer a warmstart from.

    """

    def __init__(self, identifier, train_dataset, test_dataset, results=None):
        """A MetaSample is instantiated with a unique identifier, a training dataset, a testing dataset, and the results
        of a pipeline optimization.

        Args:
            identifier (str): a unique identifier for this MetaSample.
            train_dataset (pd.DataFrame): training dataset.
            test_dataset (pd.DataFrame): test dataset.
            results (BayesianHopt): results of a Bayesian optimization run.

        """
        self._identifier = identifier
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._results = results

    @property
    def identifier(self):
        return self._identifier

    # todo: fix inconsistent naming
    @property
    def time_series(self):
        return self._train_dataset

    @property
    def test_time_series(self):
        return self._test_dataset

    @property
    def results(self):
        return self._results

    # todo: move this functionality to MetaDataset class? Since only MetaDataset actually stores metafeatures...
    def metafeatures(self, metafeature_functions):
        """Returns a vector of metafeatures of this MetaSample, using a list of metafeature functions."""
        metafeatures = pd.Series(
            data=[calc(self.time_series) for calc in metafeature_functions],
            index=[calc.__name__ for calc in metafeature_functions],
        )
        return metafeatures

    def get_best_hyperparameters(self, n):
        """Returns the best n pipeline configurations, found by a Bayesian optimization."""
        best_configs_df = self.results.sort_values(by=[("diagnostics", "mae")]).iloc[
            :n
        ]["hyperparameters"]
        best_configs = [best_configs_df.iloc[i].to_dict() for i in range(n)]
        return best_configs

    def get_best_performance(self, metric):
        """Returns the best achieved model performance according to a metric, found by a Bayesian optimization."""
        assert (
            metric in self.results["diagnostics"].columns.values
        ), "metric is not in diagnostic results"
        best_perf = self.results.sort_values(by=[("diagnostics", metric)]).iloc[0][
            ("diagnostics", "mae")
        ]
        return best_perf


class MetaDataset:
    """A MetaDataset is a collection of MetaSamples, used for warmstarting pipeline optimizations by metalearning."""

    def __init__(self, metasamples, metafeature_functions):
        """A MetaDataset is instantiated with metasamples and metafeature functions.

        Args:
            metasamples (list of MetaSample): the metasamples.
            # todo: what is documentation format for giving inputs and outputs of functions within a list?
            metafeature_functions (list of function): metafeature functions that map datasets to scalars

        """
        self.metasamples = metasamples
        self.metafeature_functions = metafeature_functions
        # todo: think about moving the metafeature calculation to the Warmstarter.
        self.metafeature_set = pd.DataFrame(
            data=[
                metasample.metafeatures(metafeature_functions)
                for metasample in tqdm(
                    self.metasamples, desc="Calculate metafeatures of metasamples"
                )
            ],
            index=[metasample.identifier for metasample in self.metasamples],
        )
