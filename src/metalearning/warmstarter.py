import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from src.metalearning.metadata import MetaSample


class Warmstarter:

    def __init__(self, metadataset, nr_configs=1, mode='warm'):
        self._metadataset = metadataset
        self._nr_configs = nr_configs
        self._mode = mode

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
        if self._mode == 'warm':
            similar_identifier = sims_diff.idxmin().values[0]
            suggestions = \
            [sample.get_best_hyperparameters(self._nr_configs) for sample in self._metadataset.metasamples if
             sample.identifier == similar_identifier][0]

        elif self._mode == 'cold':
            similar_identifier = sims_diff.idxmax().values[0]
            suggestions = \
            [sample.get_best_hyperparameters(self._nr_configs) for sample in self._metadataset.metasamples if
             sample.identifier == similar_identifier][0]

        elif self._mode == 'best_overall':
            DATA_DIR = '../..//data/metadata/raw/'
            dataset_names = ['COAST_box_17520.csv','COAST_box_8760.csv','COAST_diff_17520.csv','COAST_diff_8760.csv','EAST_box_17520.csv','EAST_box_8760.csv','EAST_diff_17520.csv','EAST_diff_8760.csv','FARWEST_box_17520.csv','FARWEST_box_8760.csv','FARWEST_diff_17520.csv','FARWEST_diff_8760.csv','NORTHC_box_17520.csv','NORTHC_box_8760.csv','NORTHC_diff_17520.csv','NORTHC_diff_8760.csv','NORTH_box_17520.csv','NORTH_box_8760.csv','NORTH_diff_17520.csv','NORTH_diff_8760.csv','SOUTHC_box_17520.csv','SOUTHC_box_8760.csv','SOUTHC_diff_17520.csv','SOUTHC_diff_8760.csv','SOUTHERN_box_17520.csv','SOUTHERN_box_8760.csv','SOUTHERN_diff_17520.csv','SOUTHERN_diff_8760.csv','WEST_box_17520.csv','WEST_box_8760.csv','WEST_diff_17520.csv','WEST_diff_8760.csv']
            dataset_names.remove(drop_index[0] + '.csv')
            datasets = [pd.read_csv(DATA_DIR + dataset_name, index_col=0, header=[0, 1]) for dataset_name in
                        dataset_names]
            all_best_10 = pd.concat(
                [dataset.sort_values(by=[('diagnostics', 'mae')]).iloc[:10]['hyperparameters'] for dataset in datasets])
            count = all_best_10.groupby(all_best_10.columns.tolist(), as_index=False).size()
            best = count.sort_values(ascending=False)[:self._nr_configs]
            best_df = pd.DataFrame(data=list(best.index.values), columns=best.index.names)
            suggestions = [best_df.iloc[i].to_dict() for i in range(5)]

        return suggestions
