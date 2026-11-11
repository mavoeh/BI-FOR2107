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
    """Return every numeric imaging feature in the assignment table."""
    return [
        name
        for name in assignments.columns
        if name not in {"subject_id", "cluster", "n_ap_volumes", "n_fm_volumes"}
        and pd.api.types.is_numeric_dtype(assignments[name])
    ]


def _feature_label(name: str) -> str:
    return _code(name) if _code(name) in EDGE_SPEC else name.replace("_", "\n")


def _feature_color(name: str) -> str:
    """Stable colors for published edges and adapted FOR2107 features."""
    code = _code(name)
    if code in EDGE_SPEC:
        return GROUP_COLORS[EDGE_SPEC[code][2]]
    lowered = name.lower()
    if "ppi" in lowered:
        return "#c26d3a"
    if "ap_" in lowered or lowered.startswith("ap"):
        return "#8064a2"
    if "fm_" in lowered or lowered.startswith("fm"):
        return "#3f8f6b"
    return "#66737a"


def _feature_group(name: str) -> str:
    lowered = name.lower()
    if "_ppi_" in lowered:
        return "Task PPI"
    if lowered.startswith("ap_"):
        return "Affective-priming activation"
    if lowered.startswith("fm_"):
        return "Face-matching activation"
    return "Published resting-state edges"


def _compact_feature_label(name: str) -> str:
    """Short labels that remain legible when all 31 features are plotted."""
    if _code(name) in EDGE_SPEC:
        return _code(name)
    tokens = name.split("_")
    task = tokens[0]
    if "_ppi_" in name:
        left, right = name.split("__", maxsplit=1)
        left_tokens = left.split("_")
        right_tokens = right.split("_")
        return (
            f"{task} {left_tokens[-2]}-{left_tokens[-1]}\n↔ {right_tokens[-2]}-{right_tokens[-1]}"
        )
    if "activation_adapted" in name:
        activation_index = tokens.index("activation")
        node = tokens[activation_index - 2 : activation_index]
        contrast = tokens[1].replace("vs", "")
        return f"{task} {contrast}\n{node[0]}-{node[1]}"
    return name.replace("_", "\n")


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
    colors = [_feature_color(feature) for feature in features]
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
                    x,
                    [_feature_label(feature) for feature in features],
                    rotation=60,
                    ha="right",
                    fontsize=6,
                )
            else:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
    legend = [
        Patch(facecolor=GROUP_COLORS["default"], label="Published resting-state edges"),
        Patch(facecolor="#8064a2", label="Affective-priming activation"),
        Patch(facecolor="#3f8f6b", label="Face-matching activation"),
        Patch(facecolor="#c26d3a", label="Task PPI"),
        Patch(facecolor="#c8cccf", label="Not FDR-significant"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005)
    )
    fig.suptitle("Split-half reliability of all imaging-feature profiles", fontweight="semibold")
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
            "All-feature FOR2107 counterpart of Tozzi Supplementary Fig. 1; the confirmatory solution is circled in red.",
            {},
        ),
        "supplementary_figure_2_simulation_silhouette": (
            simulation_silhouette_figure(assignments, evaluation, confirmatory),
            "All-feature FOR2107 counterpart of Tozzi Supplementary Fig. 2: observed mean silhouettes, multivariate-normal null p-values, and participant-level silhouettes for the confirmatory solution.",
            {},
        ),
        "supplementary_figure_3_permutation_silhouette": (
            permutation_silhouette_figure(evaluation, confirmatory),
            "All-feature FOR2107 counterpart of Tozzi Supplementary Fig. 3: observed separation and feature-permutation significance.",
            {},
        ),
        "supplementary_figure_4_stability": (
            stability_figure(loo, subsample, confirmatory),
            "All-feature FOR2107 counterpart of Tozzi Supplementary Fig. 4. Shaded bands show ±1 SD across perturbations; the confirmatory solution is circled.",
            {},
        ),
        "supplementary_figure_5_biotype_profiles": (
            _biotype_profile_figure(assignments),
            "All-feature FOR2107 counterpart of Tozzi Supplementary Fig. 5: cluster profiles with 95% confidence intervals; adapted task-derived features are shown alongside published resting-state edges.",
            {},
        ),
        "supplementary_figure_6_split_half_profiles": (
            split_figure,
            "All-feature FOR2107 counterpart of Tozzi Supplementary Fig. 6. Bars show means and 95% confidence intervals; feature colors identify resting-state edges and task-derived adaptations. Pale bands mark >0.5-SD deviations replicated in direction and magnitude across both halves.",
            {"split_tests": split_tests},
        ),
    }


def _biotype_profile_figure(assignments: pd.DataFrame) -> plt.Figure:
    """Render an all-feature profile figure in the visual grammar of Tozzi Fig. 3."""
    features = _feature_names(assignments)
    if not features:
        raise ValueError("Biotype profile figure requires imaging features")
    group_order = {
        "Published resting-state edges": 0,
        "Affective-priming activation": 1,
        "Face-matching activation": 2,
        "Task PPI": 3,
    }
    features = sorted(features, key=lambda feature: group_order[_feature_group(feature)])
    deviations = cluster_deviation_tests(assignments, features)
    clusters = sorted(assignments["cluster"].dropna().astype(int).unique())
    x = np.arange(len(features))
    height = 1.32 * len(clusters) + 2.3
    fig = plt.figure(figsize=(15.2, height), constrained_layout=False)
    grid = fig.add_gridspec(
        len(clusters),
        2,
        width_ratios=(1.05, 8.8),
        hspace=0.08,
        wspace=0.03,
        left=0.055,
        right=0.985,
        top=0.91,
        bottom=0.18,
    )
    rng = np.random.default_rng(2107)
    family_colors = {
        "Published resting-state edges": "#7389c5",
        "Affective-priming activation": "#8064a2",
        "Face-matching activation": "#3f8f6b",
        "Task PPI": "#c26d3a",
    }
    for row, cluster in enumerate(clusters):
        selected = assignments.loc[assignments["cluster"].astype(int) == cluster]
        stats = deviations.loc[deviations["cluster"].astype(int) == cluster].set_index("feature")
        profile = stats.reindex(features)
        label_ax = fig.add_subplot(grid[row, 0])
        label_ax.text(
            0.5,
            0.5,
            f"Cluster {cluster}\n(n={len(selected)})",
            ha="center",
            va="center",
            fontsize=10,
            weight="semibold",
        )
        label_ax.axis("off")
        ax = fig.add_subplot(grid[row, 1])
        for index, feature in enumerate(features):
            values = pd.to_numeric(selected[feature], errors="coerce").dropna().to_numpy()
            ax.scatter(
                np.full(len(values), index) + rng.uniform(-0.22, 0.22, len(values)),
                values,
                s=4,
                color="#8d9498",
                alpha=0.14,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
        # Keep the circuit-family color for every feature. Statistical
        # significance is reported in the accompanying statistics table, not
        # encoded by greying bars: a non-significant estimate is still a
        # meaningful observed profile value.
        colors = [_feature_color(feature) for feature in features]
        ax.bar(
            x,
            profile["mean"].to_numpy(float),
            yerr=profile["ci_95_half_width"].to_numpy(float),
            color=colors,
            width=0.72,
            edgecolor="none",
            error_kw={"ecolor": "#394147", "elinewidth": 0.65, "capsize": 1.6},
            zorder=2,
        )
        ax.axhline(0, color=INK, linewidth=0.65)
        ax.axhline(0.5, color=LINE, linewidth=0.5, linestyle="--")
        ax.axhline(-0.5, color=LINE, linewidth=0.5, linestyle="--")
        for index, feature in enumerate(features):
            if index and _feature_group(features[index - 1]) != _feature_group(feature):
                ax.axvline(index - 0.5, color="#8d969a", linewidth=0.8)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xlim(-0.75, len(features) - 0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#899197")
        ax.spines["bottom"].set_color("#899197")
        ax.grid(axis="y", color="#e5e8ea", linewidth=0.55)
        ax.set_axisbelow(True)
        if row == 0:
            starts = []
            last = None
            for index, feature in enumerate(features + ["__end__"]):
                group = _feature_group(feature) if feature != "__end__" else None
                if group != last:
                    if last is not None:
                        starts[-1][2] = index
                    if group is not None:
                        starts.append([group, index, len(features)])
                    last = group
            for group, start, end in starts:
                center = (start + end - 1) / 2
                ax.text(
                    center,
                    1.04,
                    group,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=family_colors[group],
                    weight="semibold",
                )
        if row == len(clusters) - 1:
            ax.set_xticks(
                x,
                [_compact_feature_label(feature) for feature in features],
                rotation=45,
                ha="right",
                fontsize=7,
            )
            ax.set_xlabel(
                "Features (colors identify the circuit family; error bars are 95% CIs)",
                labelpad=12,
            )
        else:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_ylabel("Reference SD", fontsize=8)
    legend = [Patch(facecolor=color, label=group) for group, color in family_colors.items()]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.56, 0.965),
        fontsize=9,
    )
    fig.suptitle(
        "Cluster profiles across all imaging features",
        x=0.53,
        y=0.995,
        fontsize=14,
        weight="semibold",
    )
    return fig
