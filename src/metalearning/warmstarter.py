import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from src.metalearning.metadata import MetaSample
from sklearn.preprocessing import StandardScaler


class Warmstarter:
    """A warmstarter suggests promising pipeline configurations based on previous tasks in the metadataset.

    # todo: should this explanation be in the 'suggest' method?
    The Euclidian distance between metafeatures calculates the similarity between the task at hand and the metasamples.
    From the most similar samples, the pipeline configurations with the best performance are joined. From this set, the
    most frequently occurring pipeline configurations are suggested to a pipeline optimization.

    """

    def __init__(
        self,
        metadataset,
        n_init_configs=1,
        n_sim_samples=1,
        n_best_per_sample=False,
        cold=False,
    ):
        """A warmstarter is instantiated with a metadataset, the number of initial configurations to suggest, the number
        of most similar metasamples to select, the number of best pipeline configurations of a metasample to
        consider, and optionally the choice for a coldstart.

        Args:
            metadataset (MetaDataset): the set of previous tasks.
            n_init_configs (int): the number of initial configurations to suggest.
            n_sim_samples (int): the number of most similar samples to include.
            n_best_per_sample (int, optional): the number of best pipeline configurations to consider per metasample.
            cold (boolean, optional): whether or not to switch on coldstart mode, defaults to False.

        """
        self._metadataset = metadataset
        self._n_init_configs = n_init_configs
        self._n_sim_samples = n_sim_samples
        if n_best_per_sample:
            self._n_best_per_sample = n_best_per_sample
        else:
            self._n_best_per_sample = n_init_configs
        self._cold = cold

    @property
    def metadataset(self):
        return self._metadataset

    @property
    def n_init_configs(self):
        return self._n_init_configs

    def suggest(self, time_series):
        """Suggests promising pipeline configurations as initialization for the pipeline optimization method.

        Args:
            time_series: the dataset of the task at hand.

        Returns:
            suggestions (list of dict): the suggested pipeline configurations.

        """
        # make a metasample
        target_sample = MetaSample("target", time_series, test_dataset=None)

        # standardize metafeatures
        df = self._metadataset.metafeature_set
        scaler = StandardScaler().fit(df)
        st_metafeature_set = pd.DataFrame(
            data=scaler.transform(df), columns=df.columns, index=df.index
        )
        st_metafeature_sample = (
            target_sample.metafeatures(self._metadataset.metafeature_functions)
            - scaler.mean_
        ) / scaler.scale_

        # calculate similarities
        if self._cold:
            sims = -cdist(
                st_metafeature_set,
                pd.DataFrame(st_metafeature_sample).T,
                metric="euclidean",
            )
        else:
            sims = cdist(
                st_metafeature_set,
                pd.DataFrame(st_metafeature_sample).T,
                metric="euclidean",
            )
        # todo: assert that MetaSamples have different metafeatures, create possibility for 1 metafeature (gives error now)
        sims_df = pd.DataFrame(data=sims, index=self._metadataset.metafeature_set.index)

        # todo: edit this functionality to -> remove the dataset at hand: difference is that different dataset could have equal metafeatures
        # remove 100% similar dataset from the samples to choose from
        drop_index = self._metadataset.metafeature_set.index[np.where(sims == 0)[0]]
        sims_diff = sims_df.drop(drop_index)

        # get initial set of hyperparameters
        suggestions = []
        sim_ids = sims_diff.nsmallest(self._n_sim_samples, sims_diff.columns).index
        for sim_id in sim_ids:
            configs = [
                pd.DataFrame(sample.get_best_hyperparameters(self._n_best_per_sample))
                for sample in self._metadataset.metasamples
                if sample.identifier == sim_id
            ]
            suggestions.extend(configs)
        interim = pd.concat(suggestions)
        count = interim.groupby(interim.columns.tolist(), as_index=False).size()
        best = count.sort_values(ascending=False)[: self._n_init_configs]
        suggestions = [dict(zip(best.index.names, values)) for values in best.index]

        return suggestions
