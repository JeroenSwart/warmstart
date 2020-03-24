import pandas as pd
from src.pipeline_optimization.bayesian_hopt import Config
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def thesis_lookup_objective(name):
    def objective(params):
        # import lookup table
        lookup_table = pd.read_csv(
            "../../data/metadata/raw/" + name + ".csv", index_col=0, header=[0, 1]
        )
        lookup_table.loc[:, ("hyperparameters", "learning_rate")] = lookup_table[
            "hyperparameters"
        ]["learning_rate"].round(13)

        idx = lookup_table.index[
            (lookup_table["hyperparameters"]["max_depth"] == params["max_depth"])
            & (
                lookup_table["hyperparameters"]["learning_rate"]
                == params["learning_rate"]
            )
            & (
                lookup_table["hyperparameters"]["min_child_weight"]
                == params["min_child_weight"]
            )
            & (lookup_table["hyperparameters"]["subsample"] == params["subsample"])
            & (lookup_table["hyperparameters"]["num_trees"] == params["num_trees"])
        ]
        result = lookup_table.iloc[idx]["diagnostics"]["mae"].squeeze()
        walltime = lookup_table.iloc[idx]["diagnostics"]["walltime"].squeeze()
        crossval = lookup_table.iloc[idx]["crossval_diag"]["mae"].squeeze()

        return result, walltime, crossval

    return objective


def thesis_search_space():
    search_space = {
        "num_trees": Config(scope=[100, 800], granularity=6, rounding=1),
        "learning_rate": Config(
            scope=[-2.5, -0.5], granularity=10, scale="log", rounding=13
        ),
        "max_depth": Config(scope=[5, 20], granularity=8, rounding=0),
        "min_child_weight": Config(scope=[5, 40], granularity=3, rounding=1),
        "subsample": Config(scope=[0.5, 1.0], granularity=3, rounding=2),
    }
    return search_space


def get_standard_dataset(dataset_name):

    # load data
    df = pd.read_csv("../../data/timeseries/raw/final_data.csv", index_col=0)

    # select the dataset
    split_name = dataset_name.split("_")
    end_name = split_name[0] + "_target_" + split_name[1]
    ex_name = split_name[0] + "_temp_" + split_name[1]
    time_based_features = ["Hour of Day", "Day of Week", "Day of Year", "Holiday"]
    data = df[[end_name, ex_name] + time_based_features].rename(
        columns={end_name: "endogenous", ex_name: "exogenous"}
    )
    dataset = data.dropna(subset=["endogenous"])[: int(split_name[2])]
    test_data = data.dropna(subset=["endogenous"])[
        int(split_name[2]) : int(split_name[2]) + 365 * 24
    ]

    return dataset, test_data


def visualize_avg_performance_single_datasets(hopt_exp, sample_ids):

    fig = make_subplots(rows=1, cols=len(sample_ids))

    for i, sample_id in enumerate(sample_ids):
        # transform to best so far dataframe
        data = hopt_exp.best_so_far[sample_id].mean(level="iterations")

        for identifier in [hopt.identifier for hopt in hopt_exp._hopts]:
            fig.add_trace(
                go.Scatter(y=data[identifier], name=identifier), row=1, col=i + 1
            )

        fig.update_layout(
            title=sample_id,
            xaxis=go.layout.XAxis(title="Iterations"),
            yaxis=go.layout.YAxis(title="MAE"),
        )

    fig.update_layout(height=600, width=1500, title_text="Subplots")

    fig.show()
