"""Generate vector figures for the 2026-07-23 preliminary-results deck."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ANALYSIS = ROOT / "derivatives" / "for2107_tozzi_external_rs" / "analysis"

NAVY = "#17324D"
BLUE = "#247BA0"
TEAL = "#20A39E"
AMBER = "#E09F3E"
RED = "#C94C4C"
GREY = "#718096"
LIGHT = "#E8EEF2"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#CBD5E0",
            "axes.linewidth": 0.8,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "pdf.fonttype": 42,
        }
    )


def generate_preliminary_results() -> None:
    evaluation = pd.read_csv(ANALYSIS / "solution_evaluation.tsv", sep="\t")
    assignments = pd.read_csv(ANALYSIS / "cluster_assignments.tsv", sep="\t")
    features = [
        "default_mode_connectivity",
        "salience_connectivity",
        "attention_connectivity",
    ]
    labels = ["Default mode", "Salience", "Attention"]
    profiles = assignments.groupby("cluster")[features].mean()
    cluster_sizes = assignments["cluster"].value_counts().sort_index()

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 3.55), gridspec_kw={"wspace": 0.32})

    axis = axes[0]
    axis.plot(
        evaluation["n_clusters"],
        evaluation["silhouette"],
        color=TEAL,
        marker="o",
        markersize=5,
        linewidth=2,
    )
    selected = evaluation.loc[evaluation["n_clusters"] == 3].iloc[0]
    axis.scatter([3], [selected["silhouette"]], s=90, facecolor="white", edgecolor=RED, linewidth=2, zorder=4)
    axis.annotate(
        "working cut: k = 3\nnull p = 0.98 | perm. p = 0.98",
        xy=(3, selected["silhouette"]),
        xytext=(3.35, 0.54),
        arrowprops={"arrowstyle": "-", "color": GREY},
        color=RED,
        fontsize=8,
    )
    axis.set(
        title="Candidate solutions",
        xlabel="Number of clusters",
        ylabel="Mean silhouette",
        xticks=evaluation["n_clusters"],
        ylim=(0.45, 0.82),
    )
    axis.text(
        0.02,
        0.04,
        "50 null + 50 permutation runs (smoke test)",
        transform=axis.transAxes,
        color=GREY,
        fontsize=7.5,
    )

    axis = axes[1]
    x = np.arange(len(profiles))
    width = 0.24
    for index, (feature, label, color) in enumerate(zip(features, labels, [BLUE, AMBER, TEAL], strict=True)):
        axis.bar(x + (index - 1) * width, profiles[feature], width, color=color, label=label)
    axis.axhline(0, color=NAVY, linewidth=0.8)
    axis.axhline(0.5, color=GREY, linewidth=0.8, linestyle="--")
    axis.axhline(-0.5, color=GREY, linewidth=0.8, linestyle="--")
    axis.set(
        title="Healthy-normalized circuit profiles",
        xlabel="Exploratory cluster",
        ylabel="Mean reference SD",
        xticks=x,
        xticklabels=[f"C{cluster}\n(n={cluster_sizes[cluster]})" for cluster in profiles.index],
        ylim=(-1.02, 0.72),
    )
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.31), fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=LIGHT, linewidth=0.7)
        axis.set_axisbelow(True)

    figure.savefig(HERE / "preliminary_results.pdf", bbox_inches="tight", transparent=True)
    figure.savefig(HERE / "preliminary_results.png", dpi=220, bbox_inches="tight", transparent=True)
    plt.close(figure)


def generate_preliminary_results_v2() -> None:
    """Create a plain-language version of the preliminary-results panel."""
    evaluation = pd.read_csv(ANALYSIS / "solution_evaluation.tsv", sep="\t")
    assignments = pd.read_csv(ANALYSIS / "cluster_assignments.tsv", sep="\t")
    features = [
        "default_mode_connectivity",
        "salience_connectivity",
        "attention_connectivity",
    ]
    labels = ["Default mode", "Salience", "Attention"]
    profiles = assignments.groupby("cluster")[features].mean()
    group_sizes = assignments["cluster"].value_counts().sort_index()

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 3.55), gridspec_kw={"wspace": 0.34})

    axis = axes[0]
    axis.plot(
        evaluation["n_clusters"],
        evaluation["silhouette"],
        color=TEAL,
        marker="o",
        markersize=5,
        linewidth=2,
    )
    selected = evaluation.loc[evaluation["n_clusters"] == 3].iloc[0]
    axis.scatter(
        [3],
        [selected["silhouette"]],
        s=90,
        facecolor="white",
        edgecolor=RED,
        linewidth=2,
        zorder=4,
    )
    axis.annotate(
        "Illustrative choice: 3 groups\nnot better than either null benchmark",
        xy=(3, selected["silhouette"]),
        xytext=(3.32, 0.54),
        arrowprops={"arrowstyle": "-", "color": GREY},
        color=RED,
        fontsize=8,
    )
    axis.set(
        title="How clearly do candidate groups separate?",
        xlabel="Number of groups",
        ylabel="Separation score",
        xticks=evaluation["n_clusters"],
        ylim=(0.45, 0.82),
    )
    axis.text(
        0.02,
        0.04,
        "Preliminary run: 50 null checks per benchmark",
        transform=axis.transAxes,
        color=GREY,
        fontsize=7.5,
    )

    axis = axes[1]
    x = np.arange(len(profiles))
    width = 0.24
    colors = [BLUE, AMBER, TEAL]
    for index, (feature, label, color) in enumerate(
        zip(features, labels, colors, strict=True)
    ):
        axis.bar(x + (index - 1) * width, profiles[feature], width, color=color, label=label)
    axis.axhline(0, color=NAVY, linewidth=0.8)
    axis.axhline(0.5, color=GREY, linewidth=0.8, linestyle="--")
    axis.axhline(-0.5, color=GREY, linewidth=0.8, linestyle="--")
    axis.set(
        title="Average circuit pattern in each group",
        xlabel="Illustrative group",
        ylabel="Difference from healthy reference",
        xticks=x,
        xticklabels=[f"Group {group}\n(n={group_sizes[group]})" for group in profiles.index],
        ylim=(-1.02, 0.72),
    )
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.31), fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=LIGHT, linewidth=0.7)
        axis.set_axisbelow(True)

    figure.savefig(HERE / "preliminary_results_v2.pdf", bbox_inches="tight", transparent=True)
    figure.savefig(
        HERE / "preliminary_results_v2.png",
        dpi=220,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(figure)


def generate_preliminary_results_v3() -> None:
    """Create the minimal, decision-focused result panel used in v3."""
    evaluation = pd.read_csv(ANALYSIS / "solution_evaluation.tsv", sep="\t")
    assignments = pd.read_csv(ANALYSIS / "cluster_assignments.tsv", sep="\t")
    features = [
        "default_mode_connectivity",
        "salience_connectivity",
        "attention_connectivity",
    ]
    labels = ["Default mode", "Salience", "Attention"]
    profiles = assignments.groupby("cluster")[features].mean()
    group_sizes = assignments["cluster"].value_counts().sort_index()

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 3.5), gridspec_kw={"wspace": 0.34})

    axis = axes[0]
    axis.plot(
        evaluation["n_clusters"],
        evaluation["silhouette"],
        color=TEAL,
        marker="o",
        markersize=5,
        linewidth=2,
    )
    selected = evaluation.loc[evaluation["n_clusters"] == 3].iloc[0]
    axis.scatter(
        [3],
        [selected["silhouette"]],
        s=90,
        facecolor="white",
        edgecolor=RED,
        linewidth=2,
        zorder=4,
    )
    axis.annotate(
        "3-group illustration\nnot stronger than chance",
        xy=(3, selected["silhouette"]),
        xytext=(3.35, 0.54),
        arrowprops={"arrowstyle": "-", "color": GREY},
        color=RED,
        fontsize=8.5,
    )
    axis.set(
        title="Candidate group separation",
        xlabel="Number of groups",
        ylabel="Separation",
        xticks=evaluation["n_clusters"],
        ylim=(0.45, 0.82),
    )

    axis = axes[1]
    x = np.arange(len(profiles))
    width = 0.24
    colors = [BLUE, AMBER, TEAL]
    for index, (feature, label, color) in enumerate(
        zip(features, labels, colors, strict=True)
    ):
        axis.bar(x + (index - 1) * width, profiles[feature], width, color=color, label=label)
    axis.axhline(0, color=NAVY, linewidth=0.8)
    axis.axhline(0.5, color=GREY, linewidth=0.8, linestyle="--")
    axis.axhline(-0.5, color=GREY, linewidth=0.8, linestyle="--")
    axis.set(
        title="Average circuit patterns",
        xlabel="Illustrative group",
        ylabel="Difference from healthy controls",
        xticks=x,
        xticklabels=[f"Group {group}\n(n={group_sizes[group]})" for group in profiles.index],
        ylim=(-1.02, 0.72),
    )
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.31), fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=LIGHT, linewidth=0.7)
        axis.set_axisbelow(True)

    figure.savefig(HERE / "preliminary_results_v3.pdf", bbox_inches="tight", transparent=True)
    figure.savefig(
        HERE / "preliminary_results_v3.png",
        dpi=220,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(figure)


if __name__ == "__main__":
    configure()
    generate_preliminary_results()
    generate_preliminary_results_v2()
    generate_preliminary_results_v3()
