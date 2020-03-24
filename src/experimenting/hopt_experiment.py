import pandas as pd
from tqdm import tqdm


class HoptExperiment:
    def __init__(self, hopts, objective=None, metadataset=None, duplicates=1):
        self._hopts = hopts
        self._duplicates = duplicates
        self._metadataset = metadataset
        self._objective = objective
        self._best_so_far = pd.DataFrame()
        self.results = None

    @property
    def best_so_far(self):
        if self._best_so_far.empty:
            best_so_far = []
            target_ids = self.results.columns.levels[0].values
            for sample_name in target_ids:
                form = self.results[sample_name].unstack([0, 1]).unstack(2)
                best_lists = [
                    [form.iloc[j][: i + 1].min() for i in form.columns]
                    for j in range(len(form))
                ]
                best_so_far.append(
                    pd.DataFrame(
                        data=best_lists, columns=form.columns, index=form.index
                    )
                    .stack(0)
                    .unstack(0)
                )
            self._best_so_far = pd.concat(best_so_far, keys=target_ids, axis=1)
        return self._best_so_far

    def run_hopt_experiment(self, target_ids):

        results = []
        samples = [
            sample
            for sample in self._metadataset.metasamples
            if sample.identifier in target_ids
        ]
        if len(target_ids) > 1:
            samples = tqdm(samples, desc="Target time series")

        for i, sample in enumerate(samples):

            time_series = sample.time_series

            for j in range(len(self._hopts)):
                self._hopts[j].objective = self._objective(sample.identifier)

            # run bayesian optimizations
            if len(target_ids) == 1:
                sample_results = [
                    [
                        hopt.run_bayesian_hopt(time_series, show_progressbar=False)
                        for n in tqdm(
                            range(self._duplicates),
                            desc=hopt.identifier + " duplicates",
                        )
                    ]
                    for hopt in self._hopts
                ]
            elif len(target_ids) > 1:
                sample_results = [
                    [
                        hopt.run_bayesian_hopt(time_series, show_progressbar=False)
                        for n in range(self._duplicates)
                    ]
                    for hopt in self._hopts
                ]

            # transform to a readable result
            df = [
                item["results"]["loss"]
                for sublist in sample_results
                for item in sublist
            ]
            indices = pd.MultiIndex.from_product(
                iterables=[
                    [hopt.identifier for hopt in self._hopts],
                    range(self._duplicates),
                ],
                names=["hopt", "duplicate_nr"],
            )
            sample_results = (
                pd.DataFrame(df, index=indices).stack().unstack(1).transpose()
            )
            sample_results = sample_results.rename_axis(columns=["hopt", "iterations"])

            # append to results
            results.append(sample_results)

        self.results = pd.concat(results, keys=target_ids, axis=1).stack(2)

        # todo: now we have to make sure that a mean is taken, s.t. all the visualizers work again
