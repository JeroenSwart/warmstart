import multiprocessing
import pandas as pd
from p_tqdm import p_umap


class HoptExperiment:
    def __init__(self, hopts, duplicates=1):
        self._hopts = hopts
        self._duplicates = duplicates
        self.results = None

    def run_hopt_experiment(self, time_series=None, num_processes=multiprocessing.cpu_count()):

        results = [p_umap(hopt.run_bayesian_hopt, [time_series]*self._duplicates, [False]*self._duplicates, num_cpus=num_processes) for hopt in self._hopts]

        df = [item['results']['loss'] for sublist in results for item in sublist]
        indices = pd.MultiIndex.from_product(
            iterables=[[hopt.identifier for hopt in self._hopts], range(self._duplicates)],
            names=['hopt', 'duplicate_nr']
        )
        results = pd.DataFrame(df, index=indices).stack().unstack(1).transpose()
        results = results.rename_axis(columns=['hopt', 'iterations'])

        return results
