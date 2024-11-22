from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from paper_analysis import get_base_color, get_figsize, save_and_close


def ranking_plot_nnunet(
    ranking_counts, ranking_medians, ranking_means, algo_order, algo_col, rank_col
):
    n_methods = ranking_counts[algo_col].nunique()
    fig, ax = plt.subplots(layout="constrained")
    offset = 0.0
    for algo_name in algo_order:
        method_counts = ranking_counts[ranking_counts[algo_col] == algo_name]
        curr_counts = method_counts.sort_values(rank_col)
        ax.bar(
            x=curr_counts[rank_col],
            height=curr_counts["norm_counts"],
            # height=1,
            bottom=offset,
            label=algo_name,
        )
        # add a vertical line at the median
        ax.vlines(
            ranking_medians.loc[algo_name],
            ymin=offset,
            ymax=offset + 0.8,
            color="black",
            linestyle="-",
            linewidth=1,
        )
        ax.vlines(
            ranking_means.loc[algo_name],
            ymin=offset,
            ymax=offset + 0.8,
            color="cyan",
            linestyle="-",
            linewidth=1,
        )
        offset += 1
    # add grid lines to plot
    ax.set_axisbelow(True)
    ax.grid(axis="both", linestyle="--")
    # remove yticklabels
    ax.set_yticklabels([])
    ax.set_xlabel("Rank")
    # add legend
    ax.legend()
    sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.05), ncol=3)
    ax.set_ylim(-0.2, n_methods + 0.2)
    return fig


def ranking_plot_blob(
    ranking_counts, ranking_medians, ranking_means, algo_order, algo_col, rank_col
):
    fig, ax = plt.subplots(
        layout="constrained", figsize=get_figsize(textwidth_factor=0.48, aspect_ratio=0.8)
    )
    offset = 0.0
    for algo_name in algo_order:
        method_counts = ranking_counts[ranking_counts[algo_col] == algo_name]
        curr_counts = method_counts.sort_values(rank_col)
        color = get_base_color(algo_name, return_idx=False)
        ax.scatter(
            x=np.ones_like(curr_counts[rank_col]) * offset,
            y=curr_counts[rank_col],
            s=curr_counts["norm_counts"] * 300,
            label=algo_name,
            c=[color] * len(curr_counts[rank_col]),
        )
        # add an x at the median for each method
        ax.plot(
            offset,
            ranking_medians.loc[algo_name],
            color="black",
            marker="x",
            markersize=10,
        )
        offset += 1
    # add grid lines to plot
    ax.set_axisbelow(True)
    ax.grid(axis="both", linestyle="--")
    # remove xticklabels
    ax.set_xticklabels([])
    ax.set_ylabel("Rank")
    # add legend
    ax.legend()
    # adapt axis limits so that all plot elements fit
    ax.set_ylim(0.6, len(algo_order) + 1 - 0.6)
    # homogenize the size of the legend artists
    for lh in ax.get_legend().legend_handles:
        lh._sizes = [10]
    sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    return fig


# This is similar to challengeR's blob plot
def visualize_ranking_distribution(
    ranking_df, algo_col="method", rank_col="ranking", style="blob", algo_order=None
):
    assert set([algo_col, rank_col]).issubset(ranking_df.columns)
    ranking_counts = (
        ranking_df.groupby(algo_col)[rank_col].value_counts().reset_index(name="count")
    )
    ranking_medians = ranking_df.groupby(algo_col)[rank_col].quantile(0.5, interpolation="linear")
    ranking_means = ranking_df.groupby(algo_col)[rank_col].mean()

    n_ranks_per_method = ranking_counts.groupby(algo_col)["count"].sum()
    assert np.all(n_ranks_per_method == n_ranks_per_method.iloc[0])
    n_ranks_per_method = n_ranks_per_method.iloc[0]
    ranking_counts["norm_counts"] = ranking_counts["count"] / n_ranks_per_method * 0.9
    # normalize counts to [0, 0.9] so that there is some white space between the bars

    if algo_order is None:
        # sort by median ranking
        algo_order = ranking_medians.sort_values(ascending=True).index.tolist()

    if style == "nnunet":
        fig = ranking_plot_nnunet(
            ranking_counts, ranking_medians, ranking_means, algo_order, algo_col, rank_col
        )
    elif style == "blob":
        fig = ranking_plot_blob(
            ranking_counts, ranking_medians, ranking_means, algo_order, algo_col, rank_col
        )
    else:
        raise ValueError(f"Unknown style: {style}")

    return fig


def compute_ranking(scores, fd_metric, higher_better):
    assert set(["bootstrap", "method", fd_metric]).issubset(set(scores.columns))
    # select one fd metric
    scores = scores[["bootstrap", "method", fd_metric]]
    # split the dataframe by bootstrap and compute a ranking for all methods
    scores["ranking"] = scores.groupby("bootstrap")[fd_metric].rank(ascending=not higher_better)
    return scores


def simulate_bootstrap_results(n_bootstraps=100):
    random_scores = {f"method{i}": np.random.randn(n_bootstraps) + (i + 1) for i in range(6)}
    df = pd.DataFrame(random_scores)
    df = df.stack().rename_axis(["bootstrap", "method"])  # -> Series
    df = pd.DataFrame(df)
    df.columns = ["aurc"]
    return df.reset_index()


def main():
    fd_metric = "aurc"
    algo_order = [
        "Ensemble + pairwise DSC",
        "MC-Dropout + pairwise DSC",
        "Ensemble + Quality regression",
        "Single network + Mahalanobis",
        "Single network + mean PE",
        "Ensemble + VAE (seg)",
    ]
    metric_name_mapper = {
        "generalized_dice": "Generalized DSC",
        "mean_dice": "Mean DSC",
        "mean_surface_dice": "Mean NSD",
    }
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="Liberation Sans",
        font_scale=0.7,
        rc={"axes.titlesize": 7},
    )

    # load results of bootstrapping evaluation
    root_dir = Path(
        "/home/m167k/Projects/segmentation_failures/gitlab_segfail/analysis/figures/expts_mar24"
    )
    all_folds_combined = []
    for fold in range(5):
        bootstrap_dir = root_dir / f"bootstrapping_fold{fold}"
        output_dir = bootstrap_dir / "figures"
        output_dir.mkdir(exist_ok=True)
        for metric_dir in bootstrap_dir.iterdir():
            if not metric_dir.is_dir():
                continue
            combined_rankings = []
            for bootstrap_csv in metric_dir.glob("*.csv"):
                df = pd.read_csv(bootstrap_csv)
                df = compute_ranking(df, fd_metric, higher_better=False)
                combined_rankings.append(df.copy())
                df["dataset"] = bootstrap_csv.stem
                df["metric"] = metric_dir.name
                all_folds_combined.append(df.copy())
                fig = visualize_ranking_distribution(df, algo_order=algo_order)
                fig.get_axes()[0].set_title(
                    f"Ranking stability: {metric_name_mapper[metric_dir.name]}"
                )
                save_and_close(fig, output_dir / f"{bootstrap_csv.stem}_{metric_dir.name}.png")
                print(output_dir / f"{bootstrap_csv.stem}_{metric_dir.name}.png")
            if len(combined_rankings) > 0:
                # combine all rankings
                fig = visualize_ranking_distribution(
                    pd.concat(combined_rankings, ignore_index=True), algo_order=algo_order
                )
                fig.get_axes()[0].set_title(
                    f"Ranking stability: {metric_name_mapper[metric_dir.name]}"
                )
                save_and_close(fig, output_dir / f"combined_{metric_dir.name}.png")
                print(output_dir / f"combined_{metric_dir.name}.png")
    # combine all rankings from all folds
    output_dir = root_dir / "bootstrapping_combined_folds_figures"
    output_dir.mkdir(exist_ok=True)
    all_folds_combined = pd.concat(all_folds_combined, ignore_index=True)
    for (dataset_name, metric_name), group_df in all_folds_combined.groupby(["dataset", "metric"]):
        fig = visualize_ranking_distribution(group_df, algo_order=algo_order)
        fig.get_axes()[0].set_title(f"Ranking stability: {metric_name_mapper[metric_name]}")
        save_and_close(fig, output_dir / f"{dataset_name}_{metric_name}.png")
        print(output_dir / f"{dataset_name}_{metric_name}.png")
    for metric_name, group_df in all_folds_combined.groupby("metric"):
        fig = visualize_ranking_distribution(group_df, algo_order=algo_order)
        fig.get_axes()[0].set_title(f"Ranking stability: {metric_name_mapper[metric_name]}")
        save_and_close(fig, output_dir / f"all_datasets_{metric_name}.png")
        print(output_dir / f"all_datasets_{metric_name}.png")
    # # TESTING or simulate random scores
    # output_dir = Path(
    #     "/home/m167k/Projects/segmentation_failures/gitlab_segfail/analysis/testing_bootstrap"
    # )
    # output_dir.mkdir(exist_ok=True)
    # df = simulate_bootstrap_results()
    # # visualize results
    # df = compute_ranking(df, fd_metric, higher_better=False)
    # fig = visualize_ranking_distribution(df)
    # fig.savefig(output_dir / "figure.png")


if __name__ == "__main__":
    main()
