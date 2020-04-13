import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_avg_ranks(hopt_exp):
    """Visualizes the ranks of the best-so-far test loss of search strategies in a HoptExperiment, per iteration of the
    search strategy. The visualizer averages the results over leave-one-out folds and duplicates of the experiment.

    Args:
        hopt_exp (experimenting.HoptExperiment): hyperoptimization experiment.

    """
    fig = go.Figure()

    data = hopt_exp.best_so_far.stack(0).rank(axis=1).mean(level="iterations")

    for identifier in [hopt.identifier for hopt in hopt_exp._hopts]:
        fig.add_trace(go.Scatter(y=data[identifier], name=identifier))

    fig.update_layout(
        xaxis=go.layout.XAxis(title="Iterations"), yaxis=go.layout.YAxis(title="Rank"),
    )

    fig.show()


def visualize_avg_performance(hopt_exp, sample_id):
    """Visualizes the best-so-far test loss of the search strategies in a HoptExperiment for one specific dataset,
    averaged over the duplicates of the experiment.

    Args:
        hopt_exp (experimenting.HoptExperiment): hyperoptimization experiment.
        sample_id (BayesianHopt.identifier): name of the dataset, also the identifier of the Bayesian hyperoptimization.

    """
    fig = go.Figure()

    # transform to best so far dataframe
    data = hopt_exp.best_so_far[sample_id].mean(level="iterations")

    for identifier in [hopt.identifier for hopt in hopt_exp._hopts]:
        fig.add_trace(go.Scatter(y=data[identifier], name=identifier))

    fig.update_layout(
        xaxis=go.layout.XAxis(title="Iterations"), yaxis=go.layout.YAxis(title="MAE"),
    )

    fig.show()


def visualize_performance_heatmap(hopt_exp, sample_id):
    """Visualizes heatmaps of the test loss with respect to the iteration number of one specific dataset, for every
    search strategy in the hyperoptimization experiment.

    Args:
        hopt_exp (experimenting.HoptExperiment): hyperoptimization experiment.
        sample_id (BayesianHopt.identifier): name of the dataset, also the identifier of the Bayesian hyperoptimization.

    """
    # todo: check for duplicates > 1
    hopt_ids = [hopt.identifier for hopt in hopt_exp._hopts]
    result = hopt_exp.results[sample_id]

    fig = make_subplots(rows=1, cols=len(hopt_ids), subplot_titles=hopt_ids)

    for j, hopt_id in enumerate(hopt_ids):
        data = result[hopt_id]
        x = list(data.index.levels[1]) * hopt_exp._duplicates
        y = data.values
        fig.add_trace(
            go.Histogram2dContour(x=x, y=y, name=hopt_id, showlegend=False),
            row=1,
            col=j + 1,
        )

    fig.update_yaxes(title_text="Mean squared error")
    fig.update_xaxes(title_text="Iterations")

    fig.show()


def visualize_perf_distribution(hopt_exp, sample_id, iteration):
    """Visualizes a boxplot of the test loss for a specific iteration of one dataset, for every search strategy in the
    hyperoptimization experiment.

    Args:
        hopt_exp (experimenting.HoptExperiment): hyperoptimization experiment.
        sample_id (BayesianHopt.identifier): name of the dataset, also the identifier of the Bayesian hyperoptimization.
        iteration (int): the iteration to visualize.

    """
    # todo: check for duplicates > 1
    fig = go.Figure()

    data = hopt_exp.results[sample_id].unstack(0).iloc[iteration].stack(1)

    for identifier in data.columns:
        fig.add_trace(
            go.Box(
                y=data[identifier],
                name=identifier,
                boxpoints="all",
                jitter=0.5,
                whiskerwidth=0.2,
                marker_size=3,
                line_width=1,
            )
        )

    fig.update_layout(yaxis=go.layout.YAxis(title="MAE"), showlegend=False)

    fig.show()


def visualize_walltime_comparison(hopt_exp, base_search, iterations):
    """Visualizes boxplots of number iterations needed, before reaching the test loss of a base search strategy at a
    certain iteration.

    Args:
        hopt_exp (experimenting.HoptExperiment): hyperoptimization experiment.
        base_search (BayesianHopt.identifier): the identifier of the compared search strategy
        iterations (int): the compared iteration

    """
    target_hopt_ids = [hopt.identifier for hopt in hopt_exp._hopts]
    target_hopt_ids.remove(base_search)
    fig = go.Figure()
    for target_hopt in target_hopt_ids:
        drop_rs_df = hopt_exp.best_so_far.stack(0).drop(columns=[base_search])
        hopt_iterations = []
        for target_sample in hopt_exp.results.columns.levels[0]:
            mean_single_search = hopt_exp.best_so_far.unstack(1)[
                (target_sample, base_search, iterations - 1)
            ].mean()
            for duplicate in range(hopt_exp._duplicates):
                one_search = drop_rs_df.unstack([0, 2])[
                    (target_hopt, duplicate, target_sample)
                ]
                if one_search.tail(1).squeeze() > mean_single_search:
                    hopt_iterations.append(iterations - 1)
                else:
                    hopt_iterations.append(
                        one_search[one_search <= mean_single_search].idxmin()
                    )
        fig.add_trace(
            go.Box(
                y=hopt_iterations,
                name=target_hopt,
                boxmean=True,
                boxpoints="all",
                jitter=0.5,
                whiskerwidth=0.2,
                marker_size=3,
                line_width=1,
            )
        )
    fig.update_layout(yaxis=go.layout.YAxis(title="Iterations"), showlegend=False)
    fig.show()
