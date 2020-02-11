import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from src.metalearning.metadata import MetaSample


class Warmstarter:

    def __init__(self, metadataset, n_init_configs=1, n_sim_samples=1, n_best_per_sample=False, cold=False):
        self._metadataset = metadataset
        self._n_init_configs = n_init_configs
        self._n_sim_samples = n_sim_samples
        if n_best_per_sample:
            self._n_best_per_sample = n_best_per_sample
        else:
            self._n_best_per_sample = n_init_configs
        self._n_best_per_sample = n_best_per_sample
        self._cold = cold

    @property
    def metadataset(self):
        return self._metadataset

    @property
    def n_init_configs(self):
        return self._n_init_configs

    def suggest(self, time_series):

        # make a metasample
        target_sample = MetaSample('target', time_series, test_dataset=None)

        # standardize metafeatures
        st_metafeature_set = self._metadataset.metafeature_set / self._metadataset.metafeature_set.max()
        st_metafeature_sample = target_sample.metafeatures(self._metadataset.metafeature_functions) / self._metadataset.metafeature_set.max()

        # calculate similarities
        if self._cold:
            sims = -cdist(st_metafeature_set, pd.DataFrame(st_metafeature_sample).T, metric='euclidean')
        else:
            sims = cdist(st_metafeature_set, pd.DataFrame(st_metafeature_sample).T, metric='euclidean')
        sims_df = pd.DataFrame(data=sims, index=self._metadataset.metafeature_set.index)

        # remove 100% similar dataset from the samples to choose from
        drop_index = self._metadataset.metafeature_set.index[np.where(sims == 0)[0]]
        sims_diff = sims_df.drop(drop_index)

        # get initial set of hyperparameters
        suggestions = []
        sim_ids = sims_diff.nsmallest(self._n_sim_samples, sims_diff.columns).index
        for sim_id in sim_ids:
            configs = [pd.DataFrame(sample.get_best_hyperparameters(self._n_best_per_sample)) for sample in
                       self._metadataset.metasamples if sample.identifier == sim_id]
            suggestions.extend(configs)
        interim = pd.concat(suggestions)
        count = interim.groupby(interim.columns.tolist(), as_index=False).size()
        best = count.sort_values(ascending=False)[:self._n_init_configs]
        suggestions = [dict(zip(best.index.names, values)) for values in best.index]

        return suggestions