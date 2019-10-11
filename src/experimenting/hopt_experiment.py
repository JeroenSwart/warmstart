import pandas as pd
from tqdm import tqdm


class HoptExperiment:
    def __init__(self, hopts, duplicates=1):
        self._hopts = hopts
        self._duplicates = duplicates
        self.results = None

    def run_hopt_experiment(self, time_series, show_progressbar=True):

        if show_progressbar:
            results = [[hopt.run_bayesian_hopt(time_series, show_progressbar=False) for i in
                        tqdm(range(self._duplicates), desc='duplicates')] for hopt in self._hopts]
        else:
            results = [[hopt.run_bayesian_hopt(time_series, show_progressbar=False) for i in range(self._duplicates)]
                       for hopt in self._hopts]

        df = [item['results']['loss'] for sublist in results for item in sublist]
        indices = pd.MultiIndex.from_product(
            iterables=[[hopt.identifier for hopt in self._hopts], range(self._duplicates)],
            names=['hopt', 'duplicate_nr']
        )
        results = pd.DataFrame(df, index=indices).stack().unstack(1).transpose()
        results = results.rename_axis(columns=['hopt', 'iterations'])

        return results
