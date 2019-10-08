import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from src.metalearning.metadata import MetaSample


class Warmstarter:

    def __init__(self, metadataset, nr_configs=1):
        self._metadataset = metadataset
        self._nr_configs = nr_configs

    @property
    def metadataset(self):
        return self._metadataset

    @property
    def nr_configs(self):
        return self._nr_configs

    def suggest(self, time_series):
        # make a metasample
        target_sample = MetaSample('target', time_series)

        # standardize metafeatures
        st_metafeature_set = self._metadataset.metafeature_set / self._metadataset.metafeature_set.max()
        st_metafeature_sample = target_sample.metafeatures / self._metadataset.metafeature_set.max()

        # calculate similarities
        sims = cdist(st_metafeature_set, pd.DataFrame(st_metafeature_sample).T, metric='euclidean')
        sims_df = pd.DataFrame(data=sims, index=self._metadataset.metafeature_set.index)

        # remove 100% similar dataset from the samples to choose from
        drop_index = self._metadataset.metafeature_set.index[np.where(sims == 0)[0]]
        sims_diff = sims_df.drop(drop_index)

        # get hyperparameters of most similar dataset
        similar_identifier = sims_diff.idxmin().values[0]
        warmstart_configs = [sample.get_best_hyperparameters(self._nr_configs) for sample in self._metadataset.metasamples if sample.identifier == similar_identifier][0]

        return warmstart_configs
