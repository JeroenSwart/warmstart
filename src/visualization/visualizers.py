import pandas as pd
import plotly.graph_objects as go


def visualize_search_performance(
    bayesian_hopt, xaxis="iterations", all_losses=False, crossvalidation=False
):
    results = bayesian_hopt.results

    # create figure
    fig = go.Figure()

    # define x-axis
    if xaxis == "iterations":
        idx = results.index
    if xaxis == "walltime":
        idx = pd.Series(
            [
                results.loc[:i, ("results", "walltime")].sum()
                for i in range(len(results))
            ]
        )

    # compute best-so-far series
    rolling_min = pd.Series(
        [results.loc[:i, ("results", "loss")].min() for i in range(len(results))]
    )
    fig.add_trace(go.Scatter(x=idx, y=rolling_min, mode="lines", name="Best so far"))

    # optionally show all computed evaluations
    if all_losses:
        loss = results.loc[:, ("results", "loss")]
        fig.add_trace(
            go.Scatter(x=idx, y=loss, mode="markers", name="Iteration result")
        )

    # optionally show the crossvalidation performance
    if crossvalidation:
        rolling_crossval = pd.Series(
            [
                results.loc[:i, ("results", "crossval")].min()
                for i in range(len(results))
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=rolling_crossval, mode="lines", name="Crossvalidation so far",
            )
        )

    # optionally show all computed evaluations of crossvalidation performance
    if all_losses & crossvalidation:
        crossval = results.loc[:, ("results", "crossval")]
        fig.add_trace(
            go.Scatter(x=idx, y=crossval, mode="markers", name="Crossvalidation result")
        )

    fig.show()