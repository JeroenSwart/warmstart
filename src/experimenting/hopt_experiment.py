import pandas as pd
from tqdm import tqdm


class HoptExperiment:
    def __init__(self, hopts, objective=None, metadataset=None, duplicates=1):
        self._hopts = hopts
        self._duplicates = duplicates
        self._metadataset = metadataset
        self._objective = objective
        self.results = None
        self.best_so_far = None

        # todo: add lazy property, calculating best_so_far AND change visualizer functions, no need to calculate there anymore :)
        # form = result.unstack(0).unstack(1)
        # best_lists = [[form.iloc[j][:i + 1].min() for i in form.columns] for j in range(len(form))]
        # best_so_far = pd.DataFrame(data=best_lists, columns=form.columns, index=form.index).stack(0).unstack([0, 2])
        # data = best_so_far.stack(1).rank(axis=1).mean(level='iterations')

    # todo: choice: input target_ids + metadataset attribute OR no metadataset attribute, test that all are equal and retrieve from a sample
    def run_hopt_experiment(self, target_ids):

        results = []

        if len(target_ids) > 1:
            target_ids = tqdm(target_ids, desc='Target time series')

        for i, dataset_name in enumerate(target_ids):

            metasample = next((sample for sample in self._metadataset.metasamples if sample.identifier == dataset_name), None)
            time_series = metasample.time_series

            for i in range(len(self._hopts)):
                self._hopts[i].objective = self._objective(dataset_name)

            # run bayesian optimizations
            if len(target_ids) == 1:
                sample_results = [[hopt.run_bayesian_hopt(time_series, show_progressbar=False) for i in
                                  tqdm(range(self._duplicates), desc=hopt.identifier + ' duplicates')] for hopt in self._hopts]
            elif len(target_ids) > 1:
                sample_results = [[hopt.run_bayesian_hopt(time_series, show_progressbar=False) for i in range(self._duplicates)]
                                 for hopt in self._hopts]

            # transform to a readable result
            df = [item['results']['loss'] for sublist in sample_results for item in sublist]
            indices = pd.MultiIndex.from_product(
                iterables=[[hopt.identifier for hopt in self._hopts], range(self._duplicates)],
                names=['hopt', 'duplicate_nr']
            )
            sample_results = pd.DataFrame(df, index=indices).stack().unstack(1).transpose()
            sample_results = sample_results.rename_axis(columns=['hopt', 'iterations'])

            # append to results
            results.append(sample_results)

        self.results = results
        # todo: now we have to make sure that a mean is taken, s.t. all the visualizers work again

        return results
