import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
                best_lists = [[form.iloc[j][:i + 1].min() for i in form.columns] for j in range(len(form))]
                best_so_far.append(
                    pd.DataFrame(data=best_lists, columns=form.columns, index=form.index).stack(0).unstack(0))
            self._best_so_far = pd.concat(best_so_far, keys=target_ids, axis=1)
        return self._best_so_far

    def run_hopt_experiment(self, target_ids):

        results = []
        samples = [sample for sample in self._metadataset.metasamples if sample.identifier in target_ids]
        if len(target_ids) > 1:
            samples = tqdm(samples, desc='Target time series')

        for i, sample in enumerate(samples):

            time_series = sample.time_series

            for j in range(len(self._hopts)):
                self._hopts[j].objective = self._objective(sample.identifier)

            # run bayesian optimizations
            if len(target_ids) == 1:
                sample_results = [[hopt.run_bayesian_hopt(time_series, show_progressbar=False) for n in
                                  tqdm(range(self._duplicates), desc=hopt.identifier + ' duplicates')] for hopt in self._hopts]
            elif len(target_ids) > 1:
                sample_results = [[hopt.run_bayesian_hopt(time_series, show_progressbar=False) for n in range(self._duplicates)]
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

        self.results = pd.concat(results, keys=target_ids, axis=1).stack(2)

        # todo: now we have to make sure that a mean is taken, s.t. all the visualizers work again

    def visualize_avg_ranks(self):

        fig = go.Figure()

        data = self.best_so_far.stack(0).rank(axis=1).mean(level='iterations')

        for identifier in [hopt.identifier for hopt in self._hopts]:
            fig.add_trace(go.Scatter(y=data[identifier], name=identifier))

        fig.update_layout(
            xaxis=go.layout.XAxis(title='Iterations'),
            yaxis=go.layout.YAxis(title='Rank')
        )

        fig.show()

    def visualize_avg_performance(self, sample_id):

        fig = go.Figure()

        # transform to best so far dataframe
        data = self.best_so_far[sample_id].mean(level='iterations')

        for identifier in [hopt.identifier for hopt in self._hopts]:
            fig.add_trace(go.Scatter(y=data[identifier], name=identifier))

        fig.update_layout(
            xaxis=go.layout.XAxis(title='Iterations'),
            yaxis=go.layout.YAxis(title='MAE')
        )

        fig.show()

    def visualize_performance_heatmap(self, sample_id):

        hopt_ids = [hopt.identifier for hopt in self._hopts]
        result = self.results[sample_id]

        fig = make_subplots(rows=1, cols=len(hopt_ids), subplot_titles=hopt_ids)

        for j, hopt_id in enumerate(hopt_ids):
            data = result[hopt_id]
            x = list(data.index.levels[1]) * self._duplicates
            y = data.values
            fig.add_trace(go.Histogram2dContour(x=x, y=y, name=hopt_id), row=1, col=j + 1)

        fig.update_yaxes(title_text="Mean squared error")
        fig.update_xaxes(title_text="Iterations")

        fig.show()

    def visualize_perf_distribution(self, sample_id, iterations):

        fig = go.Figure()

        data = self.results[sample_id].unstack(0).iloc[iterations].stack(1)

        for identifier in data.columns:
            fig.add_trace(go.Box(
                y=data[identifier],
                name=identifier,
                boxpoints='all',
                jitter=0.5,
                whiskerwidth=0.2,
                marker_size=3,
                line_width=1
            ))

        fig.update_layout(yaxis=go.layout.YAxis(title='MAE'), showlegend=False)

        fig.show()

    def visualize_walltime_comparison(self, base_search, iterations):
        target_hopt_ids = [hopt.identifier for hopt in self._hopts]
        target_hopt_ids.remove(base_search)
        fig = go.Figure()
        for target_hopt in target_hopt_ids:
            drop_rs_df = self.best_so_far.stack(0).drop(columns=[base_search])
            hopt_iterations = []
            for target_sample in self.results.columns.levels[0]:
                mean_single_search = self.best_so_far.unstack(1)[(target_sample, base_search, iterations - 1)].mean()
                for duplicate in range(self._duplicates):
                    one_search = drop_rs_df.unstack([0, 2])[(target_hopt, duplicate, target_sample)]
                    if one_search.tail(1).squeeze() > mean_single_search:
                        hopt_iterations.append(iterations-1)
                    else:
                        hopt_iterations.append(one_search[one_search <= mean_single_search].idxmin())
            fig.add_trace(go.Box(y=hopt_iterations,name=target_hopt, boxmean=True, boxpoints='all', jitter=0.5, whiskerwidth=0.2, marker_size=3, line_width=1))
        fig.update_layout(yaxis=go.layout.YAxis(title='Iterations'), showlegend=False)
        fig.show()
