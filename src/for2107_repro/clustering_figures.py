from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_samples

from .clustering import assign_by_profile_correlation, fit_clusters
from .tozzi_style import EDGE_SPEC, GROUP_COLORS, cluster_deviation_tests

INK = "#20262b"
MUTED = "#697279"
LINE = "#d5dadd"
RED = "#d9473f"


def _code(name: str) -> str:
    return name.split("_", maxsplit=1)[0]


def _feature_names(assignments: pd.DataFrame) -> list[str]:
    return [
        name
        for name in assignments.columns
        if name not in {"subject_id", "cluster"} and _code(name) in EDGE_SPEC
    ]


def _finish(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#899197")
    ax.grid(axis="y", color="#e5e8ea", linewidth=0.6)
    ax.set_axisbelow(True)


def _mark_confirmatory(ax: plt.Axes, k: int, y: float) -> None:
    ax.scatter([k], [y], s=125, facecolors="none", edgecolors=RED, linewidths=2.0, zorder=5)


def elbow_figure(evaluation: pd.DataFrame, confirmatory: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(
        evaluation["n_clusters"],
        evaluation["within_cluster_distance"],
        "o-",
        color=INK,
        markersize=4,
    )
    row = evaluation.loc[evaluation["n_clusters"] == confirmatory]
    if not row.empty:
        _mark_confirmatory(ax, confirmatory, float(row.iloc[0]["within_cluster_distance"]))
    ax.set(
        xlabel="Number of clusters",
        ylabel="Sum of within-cluster distances",
        title="Sum of distances for candidate solutions",
        xticks=evaluation["n_clusters"],
    )
    _finish(ax)
    fig.tight_layout()
    return fig


def simulation_silhouette_figure(
    assignments: pd.DataFrame,
    evaluation: pd.DataFrame,
    confirmatory: int,
) -> plt.Figure:
    features = _feature_names(assignments)
    matrix = assignments[features].to_numpy(dtype=float)
    labels = assignments["cluster"].astype(int).to_numpy()
    fit = fit_clusters(matrix, confirmatory)
    participant_silhouette = silhouette_samples(
        fit.distance_matrix,
        labels,
        metric="precomputed",
    )
    fig = plt.figure(figsize=(10.8, 6.0))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.25), hspace=0.42, wspace=0.34)
    mean_ax = fig.add_subplot(grid[0, 0])
    p_ax = fig.add_subplot(grid[1, 0])
    samples_ax = fig.add_subplot(grid[:, 1])

    mean_ax.plot(evaluation["n_clusters"], evaluation["silhouette"], "o-", color=INK)
    selected = evaluation.loc[evaluation["n_clusters"] == confirmatory].iloc[0]
    _mark_confirmatory(mean_ax, confirmatory, float(selected["silhouette"]))
    mean_ax.set(xlabel="Number of clusters", ylabel="Mean silhouette", title="a. Mean silhouette")

    p_ax.plot(evaluation["n_clusters"], evaluation["p_multinormal"], "o-", color=INK)
    p_ax.axhline(0.05, color=MUTED, linewidth=0.9, linestyle="--")
    _mark_confirmatory(p_ax, confirmatory, float(selected["p_multinormal"]))
    p_ax.set(
        xlabel="Number of clusters",
        ylabel="Simulation p-value",
        title="b. Multivariate-normal significance",
        ylim=(-0.02, 1.02),
    )

    y_lower = 8
    cluster_colors = plt.get_cmap("Greys")(np.linspace(0.35, 0.78, confirmatory))
    for index, cluster in enumerate(sorted(np.unique(labels))):
        values = np.sort(participant_silhouette[labels == cluster])
        size = len(values)
        y_upper = y_lower + size
        samples_ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            values,
            color=cluster_colors[index],
            alpha=0.95,
        )
        samples_ax.text(
            -0.035, y_lower + 0.5 * size, str(cluster), va="center", ha="right", fontsize=8
        )
        y_lower = y_upper + 8
    samples_ax.axvline(
        float(np.mean(participant_silhouette)), color=RED, linestyle="--", linewidth=1.2
    )
    samples_ax.axvline(0, color=INK, linewidth=0.7)
    samples_ax.set(
        xlabel="Participant silhouette value",
        ylabel="Participants grouped by cluster",
        title=f"c. Participant silhouettes for k={confirmatory}",
        yticks=[],
        xlim=(-0.12, 1.0),
    )
    for ax in (mean_ax, p_ax, samples_ax):
        _finish(ax)
    fig.suptitle("Simulation-based silhouette assessment", fontweight="semibold")
    return fig


def permutation_silhouette_figure(
    evaluation: pd.DataFrame,
    confirmatory: int,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    axes[0].plot(evaluation["n_clusters"], evaluation["silhouette"], "o-", color=INK)
    axes[1].plot(evaluation["n_clusters"], evaluation["p_permuted"], "o-", color=INK)
    selected = evaluation.loc[evaluation["n_clusters"] == confirmatory].iloc[0]
    _mark_confirmatory(axes[0], confirmatory, float(selected["silhouette"]))
    _mark_confirmatory(axes[1], confirmatory, float(selected["p_permuted"]))
    axes[1].annotate(
        f"p = {float(selected['p_permuted']):.5f}",
        (confirmatory, float(selected["p_permuted"])),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8,
        color=RED,
    )
    axes[1].axhline(0.05, color=MUTED, linewidth=0.9, linestyle="--")
    axes[0].set(xlabel="Number of clusters", ylabel="Mean silhouette", title="Observed separation")
    axes[1].set(
        xlabel="Number of clusters",
        ylabel="Permutation p-value",
        title="Feature-permutation significance",
        ylim=(-0.02, 1.02),
    )
    for ax in axes:
        _finish(ax)
    fig.suptitle("Permutation-based silhouette assessment", fontweight="semibold")
    fig.tight_layout()
    return fig


def stability_figure(
    loo: pd.DataFrame,
    subsample: pd.DataFrame,
    confirmatory: int,
) -> plt.Figure:
    stability = loo.merge(subsample, on="n_clusters", how="inner")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    panels = [
        (axes[0], "loo_ari_mean", "loo_ari_sd", "Leave one participant out"),
        (
            axes[1],
            "subsample_ari_mean",
            "subsample_ari_sd",
            f"Leave {int(round((1 - stability['fraction'].iloc[0]) * 100))}% out",
        ),
    ]
    x = stability["n_clusters"].to_numpy(dtype=float)
    for ax, mean_name, sd_name, title in panels:
        mean = stability[mean_name].to_numpy(dtype=float)
        sd = stability[sd_name].to_numpy(dtype=float)
        ax.fill_between(x, np.maximum(-1, mean - sd), np.minimum(1, mean + sd), color="#d9e6ec")
        ax.plot(x, mean, "o-", color=INK, markersize=4)
        row = stability.loc[stability["n_clusters"] == confirmatory].iloc[0]
        _mark_confirmatory(ax, confirmatory, float(row[mean_name]))
        ax.set(xlabel="Number of clusters", title=title, xticks=x, ylim=(-0.02, 1.02))
        _finish(ax)
    axes[0].set_ylabel("Adjusted Rand index")
    fig.suptitle("Cluster stability under participant perturbation", fontweight="semibold")
    fig.tight_layout()
    return fig


def _profile_correlations(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[np.corrcoef(first_row, second_row)[0, 1] for second_row in second] for first_row in first]
    )


def _split_half_assignments(
    assignments: pd.DataFrame,
    *,
    confirmatory: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _feature_names(assignments)
    matrix = assignments[features].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    first_indices, second_indices = np.array_split(rng.permutation(len(matrix)), 2)
    first_fit = fit_clusters(matrix[first_indices], confirmatory)
    first_profiles = np.vstack(
        [
            matrix[first_indices][first_fit.labels == cluster].mean(axis=0)
            for cluster in range(1, confirmatory + 1)
        ]
    )
    whole_profiles = np.vstack(
        [
            matrix[assignments["cluster"].astype(int).to_numpy() == cluster].mean(axis=0)
            for cluster in range(1, confirmatory + 1)
        ]
    )
    rows, columns = linear_sum_assignment(-_profile_correlations(first_profiles, whole_profiles))
    label_map = {int(row + 1): int(column + 1) for row, column in zip(rows, columns, strict=True)}
    second_labels = assign_by_profile_correlation(matrix[second_indices], first_profiles)

    first = assignments.iloc[first_indices][["subject_id", *features]].copy()
    first["cluster"] = [label_map[int(value)] for value in first_fit.labels]
    second = assignments.iloc[second_indices][["subject_id", *features]].copy()
    second["cluster"] = [label_map[int(value)] for value in second_labels]
    return first[["subject_id", "cluster", *features]], second[["subject_id", "cluster", *features]]


def split_half_profile_figure(
    assignments: pd.DataFrame,
    *,
    confirmatory: int,
    seed: int,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    features = _feature_names(assignments)
    first, second = _split_half_assignments(assignments, confirmatory=confirmatory, seed=seed)
    panels = {
        "Entire sample": assignments,
        "Random split half 1": first,
        "Random split half 2": second,
    }
    tests = {name: cluster_deviation_tests(data, features) for name, data in panels.items()}
    fig, axes = plt.subplots(
        confirmatory,
        3,
        figsize=(13.0, 1.62 * confirmatory + 1.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x = np.arange(len(features))
    colors = [GROUP_COLORS[EDGE_SPEC[_code(feature)][2]] for feature in features]
    rng = np.random.default_rng(seed)
    whole_means = assignments.groupby("cluster")[features].mean()
    first_means = first.groupby("cluster")[features].mean()
    second_means = second.groupby("cluster")[features].mean()

    for row, cluster in enumerate(range(1, confirmatory + 1)):
        replicated = np.asarray(
            [
                abs(whole_means.loc[cluster, feature]) >= 0.5
                and abs(first_means.loc[cluster, feature]) >= 0.5
                and abs(second_means.loc[cluster, feature]) >= 0.5
                and np.sign(whole_means.loc[cluster, feature])
                == np.sign(first_means.loc[cluster, feature])
                == np.sign(second_means.loc[cluster, feature])
                for feature in features
            ]
        )
        for column, (panel_name, data) in enumerate(panels.items()):
            ax = axes[row, column]
            selected = data.loc[data["cluster"].astype(int) == cluster]
            test = (
                tests[panel_name]
                .loc[tests[panel_name]["cluster"].astype(int) == cluster]
                .set_index("feature")
                .reindex(features)
            )
            for index, feature in enumerate(features):
                values = pd.to_numeric(selected[feature], errors="coerce").dropna().to_numpy()
                ax.scatter(
                    np.full(len(values), index) + rng.uniform(-0.28, 0.28, len(values)),
                    values,
                    s=3,
                    color="#777d81",
                    alpha=0.12,
                    linewidths=0,
                    rasterized=True,
                )
                if replicated[index]:
                    ax.axvspan(
                        index - 0.47, index + 0.47, color=colors[index], alpha=0.10, zorder=0
                    )
            bar_colors = [
                colors[index] if bool(test.loc[feature, "significant_fdr"]) else "#c8cccf"
                for index, feature in enumerate(features)
            ]
            ax.bar(
                x,
                test["mean"],
                yerr=test["ci_95_half_width"],
                width=0.66,
                color=bar_colors,
                edgecolor="none",
                error_kw={"ecolor": INK, "elinewidth": 0.7, "capsize": 1.5},
                zorder=2,
            )
            ax.axhline(0, color=INK, linewidth=0.55)
            ax.axhline(0.5, color=LINE, linewidth=0.5, linestyle="--")
            ax.axhline(-0.5, color=LINE, linewidth=0.5, linestyle="--")
            ax.set_ylim(-5, 5)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(panel_name, fontweight="semibold")
            if column == 0:
                ax.set_ylabel(f"Cluster {cluster}\nreference SD")
            if row == confirmatory - 1:
                ax.set_xticks(
                    x, [_code(feature) for feature in features], rotation=60, ha="right", fontsize=7
                )
            else:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
    legend = [
        Patch(facecolor=GROUP_COLORS["default"], label="Default mode"),
        Patch(facecolor=GROUP_COLORS["attention"], label="Attention"),
        Patch(facecolor=GROUP_COLORS["salience"], label="Salience"),
        Patch(facecolor="#c8cccf", label="Not FDR-significant"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005)
    )
    fig.suptitle("Split-half reliability of resting-state cluster profiles", fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig, tests


def supplementary_clustering_figures(
    assignments: pd.DataFrame,
    evaluation: pd.DataFrame,
    loo: pd.DataFrame,
    subsample: pd.DataFrame,
    *,
    confirmatory: int,
    seed: int,
) -> dict[str, tuple[plt.Figure, str, dict[str, Any]]]:
    split_figure, split_tests = split_half_profile_figure(
        assignments,
        confirmatory=confirmatory,
        seed=seed,
    )
    return {
        "supplementary_figure_1_elbow": (
            elbow_figure(evaluation, confirmatory),
            "Rest-only counterpart of Tozzi Supplementary Fig. 1; the confirmatory solution is circled in red.",
            {},
        ),
        "supplementary_figure_2_simulation_silhouette": (
            simulation_silhouette_figure(assignments, evaluation, confirmatory),
            "Rest-only counterpart of Tozzi Supplementary Fig. 2: observed mean silhouettes, multivariate-normal null p-values, and participant-level silhouettes for the confirmatory solution.",
            {},
        ),
        "supplementary_figure_3_permutation_silhouette": (
            permutation_silhouette_figure(evaluation, confirmatory),
            "Rest-only counterpart of Tozzi Supplementary Fig. 3: observed separation and feature-permutation significance.",
            {},
        ),
        "supplementary_figure_4_stability": (
            stability_figure(loo, subsample, confirmatory),
            "Rest-only counterpart of Tozzi Supplementary Fig. 4. Shaded bands show ±1 SD across perturbations; the confirmatory solution is circled.",
            {},
        ),
        "supplementary_figure_6_split_half_profiles": (
            split_figure,
            "Rest-only counterpart of Tozzi Supplementary Fig. 6. Bars show means and unadjusted 95% confidence intervals; circuit colors require BH-FDR significance within each panel. Pale bands mark >0.5-SD deviations replicated in direction and magnitude across both halves.",
            {"split_tests": split_tests},
        ),
    }
