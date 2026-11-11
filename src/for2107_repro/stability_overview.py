"""Two overview figures for the split-half stability analysis.

`results_overview` shows the results at a glance from the saved tables;
`method_overview` explains the analysis with schematic toy data.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stability_figures import COSINE, GRID, STYLE, SURFACE, TEXT, TEXT_MUTED, read_table

INDEPENDENT = {
    "edges_ap vs edges_fm (tozzi_edges)": "AP task  vs  FM task",
    "edges_apfm vs edges_rest (tozzi_edges)": "both tasks  vs  rest scan",
    "edges_ap vs edges_rest (tozzi_edges)": "AP task  vs  rest scan",
    "edges_fm vs edges_rest (tozzi_edges)": "FM task  vs  rest scan",
}
SHORT = {"null": "No subgroups", "observed": "FOR2107", "planted": "Tozzi's biotypes"}


def _panel_label(
    axis: plt.Axes, letter: str, title: str, subtitle: str = "", raise_by: float = 0
) -> None:
    """Bold letter and question above the axes, grey subtitle beneath them."""
    for text, x, y, size, weight, color, ha in (
        (letter, -14, 30, 14, "bold", TEXT, "right"),
        (title, 0, 30, 10.5, "bold", TEXT, "left"),
        (subtitle, 0, 14, 8.5, "normal", TEXT_MUTED, "left"),
    ):
        if text:
            axis.annotate(
                text,
                (0, 1),
                xycoords="axes fraction",
                xytext=(x, y + raise_by),
                textcoords="offset points",
                fontsize=size,
                fontweight=weight,
                va="bottom",
                ha=ha,
                color=color,
            )


def _clean(axis: plt.Axes, grid: str = "x") -> None:
    axis.grid(axis=grid, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)


def _distributions(
    axis: plt.Axes, values: dict[str, pd.Series], bins: np.ndarray, observed_line: bool
) -> None:
    """Histograms of the simulated (and observed) values with labelled medians."""
    _clean(axis, "y")
    top = 0
    for source in ("null", "planted", "observed"):
        if source not in values or (source == "observed" and observed_line):
            continue
        style = STYLE[source]
        counts, _, _ = axis.hist(
            values[source], bins=bins, color=style["color"], alpha=0.3, histtype="stepfilled"
        )
        axis.hist(values[source], bins=bins, color=style["color"], histtype="step", linewidth=1.4)
        top = max(top, counts.max())
    for source in ("null", "observed", "planted"):
        if source not in values:
            continue
        style = STYLE[source]
        median = float(np.median(values[source]))
        axis.vlines(
            median,
            0,
            top * 1.04,
            color=style["color"],
            linewidth=2.2 if source == "observed" else 1.6,
        )
    axis.set_ylim(0, top * 1.55)
    # one row of labels above the histograms, pushed apart so they never
    # overlap, each joined to its median line by a leader
    lower, upper = bins[0], bins[-1]
    # labels are about 85 px wide; convert that to data units for this panel
    width_px = axis.get_window_extent().width or 300
    gap = max(0.2, 95 / width_px) * (upper - lower)
    order = sorted(values, key=lambda source: float(np.median(values[source])))
    medians = [float(np.median(values[source])) for source in order]
    positions = list(medians)
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + gap)
    overshoot = positions[-1] - (upper - 0.5 * gap)
    if overshoot > 0:
        positions = [position - overshoot for position in positions]
        for index in range(len(positions) - 2, -1, -1):
            positions[index] = min(positions[index], positions[index + 1] - gap)
    for source, median, position in zip(order, medians, positions, strict=True):
        color = STYLE[source]["color"] if source != "null" else TEXT_MUTED
        axis.annotate(
            f"{SHORT[source]}\n{median:.2f}",
            (median, top * 1.04),
            xytext=(position, top * 1.3),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
            fontweight="bold" if source == "observed" else "normal",
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.8,
                "shrinkA": 1,
                "shrinkB": 0,
            },
        )


def results_overview(root: Path, path: Path) -> Path:
    root = Path(root)
    splits = read_table(root / "split_half_splits.tsv")
    summary = read_table(root / "split_half_summary.tsv").drop_duplicates("feature_set")
    matrices = read_table(root / "mean_cosine_matrices.tsv")
    cross = read_table(root / "cross_set_draws.tsv")
    tozzi = read_table(root / "tozzi_match_draws.tsv")

    figure = plt.figure(figsize=(13, 11))
    outer = figure.add_gridspec(
        3, 1, height_ratios=[1, 1, 0.36], hspace=0.62, top=0.88, bottom=0.02, left=0.07, right=0.97
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1.3], wspace=0.18)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)

    # a: split-half agreement, Tozzi's 31 features
    axis = figure.add_subplot(top[0])
    tozzi31 = splits.loc[splits.feature_set == "tozzi_matched"]
    _distributions(
        axis,
        {source: tozzi31.loc[tozzi31.source == source, "ari_assignment"] for source in STYLE},
        np.linspace(0, 0.45, 37),
        observed_line=False,
    )
    axis.set_xlabel("Agreement between the two halves (ARI; 0 = chance, 1 = identical)")
    axis.set_ylabel("Random splits (of 100)")
    _panel_label(
        axis,
        "a",
        "Do the same patients group together again?",
        "Tozzi's 31 features · 100 random 50/50 splits of the patients",
    )

    # b: average matched 6 x 6 cosine matrices
    inner = top[1].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.07], wspace=0.1)
    image = None
    for column, source in enumerate(("null", "observed", "planted")):
        axis = figure.add_subplot(inner[0, column])
        cells = matrices.loc[
            (matrices.feature_set == "tozzi_matched") & (matrices.source == source)
        ]
        values = np.full((6, 6), np.nan)
        values[cells.row_rank - 1, cells.column_rank - 1] = cells.cosine
        image = axis.imshow(values, vmin=-1, vmax=1, cmap=COSINE)
        for index in range(6):
            value = values[index, index]
            axis.text(
                index,
                index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=SURFACE if abs(value) > 0.55 else TEXT,
            )
        axis.set_xticks(range(6), range(1, 7), fontsize=7.5)
        axis.set_yticks(range(6), range(1, 7) if column == 0 else [], fontsize=7.5)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(
            SHORT[source],
            fontsize=9.5,
            color=STYLE[source]["color"] if source != "null" else TEXT_MUTED,
            fontweight="bold",
        )
        if column == 0:
            axis.set_ylabel("half-A cluster (largest first)", fontsize=8.5)
            _panel_label(
                axis,
                "b",
                "Do the same cluster profiles come back?",
                "Similarity of each half-A cluster with its match in half B (diagonal) and the others",
                raise_by=14,
            )
        axis.set_xlabel("matched half-B cluster", fontsize=8.5)
    colorbar = figure.colorbar(image, cax=figure.add_subplot(inner[0, 3]))
    colorbar.set_label("profile similarity (cosine)", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5)
    colorbar.outline.set_visible(False)

    # c: every feature set
    axis = figure.add_subplot(bottom[0])
    _clean(axis, "both")
    gain = summary.assign(
        sensitivity=summary.ari_assignment_planted - summary.ari_assignment_null,
        signal=summary.ari_assignment_observed - summary.ari_assignment_null,
    )
    limit = 0.27
    # feature sets where planted biotypes barely beat no subgroups
    axis.axvspan(-0.04, 0.05, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    axis.text(
        -0.035,
        0.262,
        "test cannot see\nTozzi's biotypes",
        fontsize=7.5,
        color=TEXT_MUTED,
        va="top",
        ha="left",
    )
    axis.plot([-0.04, limit], [-0.04, limit], color=STYLE["planted"]["color"], linewidth=1.5)
    axis.axhline(0, color=STYLE["null"]["color"], linewidth=1.5)
    axis.text(
        0.15,
        0.163,
        "if Tozzi's biotypes were present",
        color=STYLE["planted"]["color"],
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=45,
        rotation_mode="anchor",
    )
    axis.text(
        limit + 0.004,
        0.0,
        "no\nsubgroups",
        color=TEXT_MUTED,
        ha="left",
        va="center",
        fontsize=8,
        clip_on=False,
    )
    families = (
        (
            "from Tozzi's 31",
            gain.feature_set.str.startswith("tozzi") | (gain.pool == "tozzi_matched"),
            "o",
        ),
        (
            "broader (up to 285)",
            ~(gain.feature_set.str.startswith("tozzi") | (gain.pool == "tozzi_matched")),
            "s",
        ),
    )
    for name, mask, marker in families:
        color = STYLE["observed"]["color"]
        axis.plot(
            gain.loc[mask, "sensitivity"],
            gain.loc[mask, "signal"],
            linestyle="",
            marker=marker,
            markersize=5.5,
            color=color,
            markerfacecolor=color if marker == "o" else "none",
            alpha=0.8,
            label=name,
        )
    star = gain.loc[gain.feature_set == "tozzi_matched"].iloc[0]
    axis.annotate(
        "Tozzi's 31",
        (star.sensitivity, star.signal),
        xytext=(star.sensitivity - 0.06, 0.09),
        fontsize=8,
        color=TEXT,
        arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "linewidth": 0.8},
    )
    axis.set_xlim(-0.04, limit)
    axis.set_ylim(-0.04, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(
        "How much Tozzi's biotypes would raise agreement\n(sensitivity of the test, ARI)"
    )
    axis.set_ylabel("How much FOR2107 raises agreement (ARI)")
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.83),
        title="feature sets",
        title_fontsize=8,
        alignment="left",
        frameon=False,
        fontsize=8,
        handletextpad=0.2,
        borderaxespad=0.3,
    )
    _panel_label(
        axis, "c", "Any feature set with subgroups?", "106 feature sets · relative to no subgroups"
    )

    # d: independent measurements of the same edges
    axis = figure.add_subplot(bottom[1])
    _clean(axis)
    for row, (name, label) in enumerate(INDEPENDENT.items()):
        group = cross.loc[cross.comparison == name]
        for source, offset in (("null", 0.1), ("planted", 0.3)):
            values = group.loc[group.source == source, "ari"].dropna()
            low, mid, high = np.percentile(values, [2.5, 50, 97.5])
            style = STYLE[source]
            axis.hlines(row + offset, low, high, color=style["color"], linewidth=3, alpha=0.5)
            axis.plot(
                mid,
                row + offset,
                marker=style["marker"],
                color=style["color"],
                markersize=6,
                markeredgecolor=SURFACE,
            )
        observed = group.loc[group.source == "observed", "ari"].iloc[0]
        axis.plot(
            observed,
            row - 0.1,
            marker="o",
            color=STYLE["observed"]["color"],
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=4,
        )
        axis.text(0.003, row - 0.36, label, fontsize=8.5, color=TEXT, va="bottom", ha="left")
    axis.set_yticks([])
    axis.set_ylim(len(INDEPENDENT) - 0.45, -0.75)
    axis.set_xlim(-0.005, 0.16)
    axis.spines["left"].set_visible(False)
    axis.axvline(0, color=TEXT_MUTED, linewidth=0.6)
    axis.set_xlabel("Agreement of the two groupings (ARI)\nbars: middle 95% of 100 simulations")
    _panel_label(
        axis, "d", "Same patients, measured twice", "Tozzi's 14 connections from separate scans"
    )

    # e: resemblance to Tozzi's six biotypes
    axis = figure.add_subplot(bottom[2])
    draws = tozzi.loc[tozzi.feature_set == "tozzi_matched"]
    _distributions(
        axis,
        {source: draws.loc[draws.source == source, "mean_matched"] for source in STYLE},
        np.linspace(0.2, 0.9, 36),
        observed_line=True,
    )
    axis.set_xlabel(
        "Similarity of the six cluster profiles\nwith Tozzi's six biotypes (mean cosine)"
    )
    axis.set_ylabel("Simulations (of 100)")
    _panel_label(
        axis, "e", "Do the clusters look like Tozzi's?", "Tozzi's 31 features · all 701 patients"
    )

    handles = [
        plt.Line2D(
            [],
            [],
            color=STYLE[s]["color"],
            marker=STYLE[s]["marker"],
            linestyle="",
            markersize=9,
            label=STYLE[s]["label"],
        )
        for s in ("observed", "null", "planted")
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=10.5,
    )

    caption = figure.add_subplot(outer[2])
    caption.axis("off")
    caption.text(
        0,
        1,
        "\n".join(textwrap.wrap(RESULTS_CAPTION, 178)),
        va="top",
        ha="left",
        fontsize=8.5,
        color=TEXT,
        linespacing=1.4,
    )
    figure.savefig(path, bbox_inches="tight", pad_inches=0.25, dpi=200)
    plt.close(figure)
    return path


RESULTS_CAPTION = (
    "Figure 2 | FOR2107 patients show no reproducible subgroups. Blue: FOR2107 patients with depression and/or "
    "anxiety (n = 701). Gray: simulated patients without subgroups but with the same means, spreads and feature "
    "correlations. Orange: simulated patients carrying Tozzi's six biotypes at the strength calibrated to his "
    "study. a, The patients were split at random into two halves 100 times; each half was clustered into six "
    "groups, and every half-B patient was also assigned to the most similar half-A group. ARI measures how well "
    "the two assignments agree. FOR2107 agrees as little as data without subgroups and far less than data with "
    "Tozzi's biotypes. b, Average similarity (cosine, 1 = identical profile) of each half-A cluster with its "
    "best-matching half-B cluster (diagonal) and with the other clusters (off-diagonal). c, One point per feature "
    "set: x shows how much Tozzi's biotypes would raise the agreement in a above the no-subgroup level (the "
    "sensitivity of the test), y how much FOR2107 raises it. FOR2107 stays at zero for every feature set, even "
    "where the test is most sensitive. d, Clustering the same patients with the same 14 connections measured in "
    "different scans gives unrelated groupings. e, FOR2107's cluster profiles resemble Tozzi's biotypes no more "
    "than clusters cut from data without subgroups do."
)


# --- method schematic --------------------------------------------------------------

CLUSTER_COLORS = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _schematic(axis: plt.Axes) -> plt.Axes:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    return axis


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str = TEXT,
    fill: str = SURFACE,
    size: float = 8.5,
) -> None:
    from matplotlib.patches import FancyBboxPatch

    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.2,
        )
    )
    axis.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color)


def _arrow(
    axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = TEXT_MUTED
) -> None:
    axis.annotate(
        "",
        end,
        start,
        arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.4, "mutation_scale": 12},
    )


def _toy_profiles(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """2-D toy 'profile space' and six k-means groups, for illustration only."""
    from sklearn.cluster import KMeans

    points = rng.multivariate_normal([0, 0], [[1.0, 0.45], [0.45, 0.7]], size=n)
    labels = KMeans(6, n_init=10, random_state=0).fit_predict(points)
    order = np.argsort(-np.bincount(labels))
    return points, np.argsort(order)[labels]


def method_overview(
    path: Path,
    n_patients: int = 701,
    n_features: int = 285,
    n_tozzi: int = 31,
    splits: int = 100,
    n_sets: int = 106,
) -> Path:
    rng = np.random.default_rng(3)
    figure = plt.figure(figsize=(14.5, 10.5))
    outer = figure.add_gridspec(
        3, 1, height_ratios=[1, 1, 0.3], hspace=0.5, top=0.93, bottom=0.02, left=0.03, right=0.98
    )
    top = outer[0].subgridspec(1, 5, width_ratios=[1.05, 0.7, 1.1, 0.95, 0.85], wspace=0.32)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[1.15, 1, 1.2], wspace=0.28)

    # a: feature table and feature sets
    axis = _schematic(figure.add_subplot(top[0]))
    _panel_label(axis, "a", "Choose features", f"{n_sets} feature sets from {n_features} features")
    table = rng.normal(0, 1, (22, 30))
    table = (table + np.roll(table, 1, axis=1)) / 1.4
    axis.imshow(
        table, extent=(0.1, 0.95, 0.42, 0.93), cmap=COSINE, vmin=-2.5, vmax=2.5, aspect="auto"
    )
    axis.text(
        0.06,
        0.675,
        f"{n_patients} patients",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.5,
        color=TEXT,
    )
    tozzi_end = 0.1 + 0.85 * 9 / 30
    axis.plot([0.1, tozzi_end - 0.005], [0.955, 0.955], color=TEXT, linewidth=2)
    axis.text(
        0.1,
        0.97,
        f"Tozzi's {n_tozzi}",
        ha="left",
        va="bottom",
        fontsize=8,
        color=TEXT,
        fontweight="bold",
    )
    axis.plot([tozzi_end + 0.005, 0.95], [0.955, 0.955], color=TEXT_MUTED, linewidth=2)
    axis.text(
        0.95,
        0.97,
        f"{n_features - n_tozzi} further",
        ha="right",
        va="bottom",
        fontsize=8,
        color=TEXT_MUTED,
    )
    axis.text(
        0.52,
        0.385,
        "z-scores vs healthy controls (red low, blue high)",
        ha="center",
        va="top",
        fontsize=7.5,
        color=TEXT_MUTED,
    )
    examples = (
        ("Tozzi's 31", np.arange(30) < 9),
        ("all features", np.ones(30, bool)),
        ("one network out", ~((np.arange(30) >= 14) & (np.arange(30) < 20))),
        ("random subset", rng.random(30) < 0.5),
    )
    axis.text(
        0.08, 0.31, "e.g.", ha="right", va="center", fontsize=7.5, color=TEXT_MUTED, style="italic"
    )
    for row, (name, used) in enumerate(examples):
        y = 0.25 - row * 0.07
        axis.text(0.08, y, name, ha="right", va="center", fontsize=7.5, color=TEXT)
        for column, flag in enumerate(used):
            x = 0.1 + 0.85 * column / 30
            axis.add_patch(
                plt.Rectangle(
                    (x, y - 0.02), 0.85 / 30 * 0.8, 0.04, color=TEXT if flag else GRID, linewidth=0
                )
            )

    # b: random split
    axis = _schematic(figure.add_subplot(top[1]))
    _panel_label(axis, "b", "Split at random", f"repeated {splits}×")
    half = rng.permutation(np.repeat([0, 1], 35))
    for index, group in enumerate(half):
        x, y = 0.12 + 0.125 * (index % 7), 0.9 - 0.068 * (index // 7)
        axis.plot(
            x,
            y,
            marker="o",
            markersize=6.5,
            color=TEXT if group == 0 else SURFACE,
            markeredgecolor=TEXT,
            markeredgewidth=1,
        )
    axis.plot(0.1, 0.1, marker="o", markersize=7, color=TEXT)
    axis.text(0.17, 0.1, "half A", va="center", fontsize=8.5)
    axis.plot(0.55, 0.1, marker="o", markersize=7, color=SURFACE, markeredgecolor=TEXT)
    axis.text(0.62, 0.1, "half B", va="center", fontsize=8.5)
    axis.text(
        0.5, 0.19, f"{n_patients // 2} patients each", ha="center", fontsize=8, color=TEXT_MUTED
    )

    # c: cluster each half
    inner = top[2].subgridspec(1, 2, wspace=0.08)
    halves = []
    for column, name in enumerate(("half A", "half B")):
        axis = figure.add_subplot(inner[column])
        points, labels = _toy_profiles(rng, 120)
        halves.append((points, labels))
        for cluster in range(6):
            members = points[labels == cluster]
            axis.scatter(
                members[:, 0], members[:, 1], s=9, color=CLUSTER_COLORS[cluster], linewidths=0
            )
            centre = members.mean(axis=0)
            axis.text(
                *centre,
                str(cluster + 1),
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=TEXT,
                bbox={
                    "boxstyle": "circle,pad=0.15",
                    "facecolor": SURFACE,
                    "edgecolor": CLUSTER_COLORS[cluster],
                },
            )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(GRID)
        axis.set_xlabel(name, fontsize=8.5)
        if column == 0:
            _panel_label(axis, "c", "Cluster each half into 6", "cluster numbers are arbitrary")

    # d: compare cluster profiles (cosine)
    axis = figure.add_subplot(top[3])
    ideal = np.full((6, 6), -0.15) + rng.normal(0, 0.05, (6, 6))
    np.fill_diagonal(ideal, [0.85, 0.8, 0.75, 0.7, 0.6, 0.5])
    axis.imshow(ideal, cmap=COSINE, vmin=-1, vmax=1)
    axis.set_anchor("N")
    for index in range(6):
        axis.add_patch(
            plt.Rectangle(
                (index - 0.5, index - 0.5), 1, 1, fill=False, edgecolor=TEXT, linewidth=1.2
            )
        )
    axis.set_xticks(range(6), range(1, 7), fontsize=7.5)
    axis.set_yticks(range(6), range(1, 7), fontsize=7.5)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_ylabel("cluster in half A", fontsize=8)
    axis.set_xlabel("its best match in half B", fontsize=8)
    _panel_label(axis, "d", "Do the profiles come back?", "similarity of average profiles")
    axis.text(
        2.5,
        6.9,
        "cosine: +1 same, 0 unrelated, −1 opposite\ndiagonal = matched pairs, rest = unmatched\nseparation = matched − unmatched",
        ha="center",
        va="top",
        fontsize=8,
        color=TEXT,
        linespacing=1.35,
    )

    # e: do the same patients group together (ARI)
    axis = _schematic(figure.add_subplot(top[4]))
    _panel_label(axis, "e", "Same patients together?", "two labels per half-B patient")
    axis.text(0.5, 0.88, "own\ncluster\nin B", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    axis.text(
        0.8, 0.88, "closest\nhalf-A\nprofile", ha="center", va="bottom", fontsize=7.5, color=TEXT
    )
    own = [0, 0, 1, 2, 2, 3, 4, 5]
    nearest = [0, 0, 1, 1, 2, 3, 4, 2]
    for row, (first, second) in enumerate(zip(own, nearest, strict=True)):
        y = 0.83 - row * 0.075
        axis.text(
            0.0, y, f"patient {row + 1}", ha="left", va="center", fontsize=7, color=TEXT_MUTED
        )
        for x, cluster in ((0.5, first), (0.8, second)):
            axis.add_patch(
                plt.Rectangle(
                    (x - 0.07, y - 0.025), 0.14, 0.05, color=CLUSTER_COLORS[cluster], linewidth=0
                )
            )
        axis.text(
            0.98,
            y,
            "✓" if first == second else "✗",
            ha="center",
            va="center",
            fontsize=9,
            color=TEXT if first == second else "#d03b3b",
        )
    axis.text(
        0.5,
        0.12,
        "ARI = agreement of the two columns,\ncorrected for chance\n0 = chance, 1 = identical",
        ha="center",
        va="center",
        fontsize=8,
        color=TEXT,
    )

    # e: simulated references
    inner = bottom[0].subgridspec(1, 2, wspace=0.08)
    for column, source in enumerate(("null", "planted")):
        axis = figure.add_subplot(inner[column])
        if source == "null":
            points = rng.multivariate_normal([0, 0], [[1.0, 0.45], [0.45, 0.7]], size=300)
            note = "one smooth cloud"
        else:
            centres = np.array(
                [[-1.5, -0.8], [1.4, 0.9], [-0.1, 1.5], [0.6, -1.4], [-1.9, 1.2], [2.0, -0.9]]
            )
            sizes = np.array([0.32, 0.21, 0.2, 0.19, 0.05, 0.03])
            truth = rng.choice(6, size=300, p=sizes / sizes.sum())
            points = centres[truth] + rng.normal(0, 0.42, (300, 2))
            note = "six overlapping groups"
        axis.scatter(
            points[:, 0], points[:, 1], s=8, color=STYLE[source]["color"], alpha=0.6, linewidths=0
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim(-3.4, 3.4)
        axis.set_ylim(-3, 3)
        axis.text(0, 2.75, note, ha="center", va="top", fontsize=8, color=TEXT_MUTED)
        for spine in axis.spines.values():
            spine.set_color(GRID)
        axis.set_xlabel(
            SHORT[source],
            fontsize=9,
            color=STYLE[source]["color"] if source != "null" else TEXT_MUTED,
            fontweight="bold",
        )
        if column == 0:
            _panel_label(
                axis,
                "f",
                "Repeat b–e on simulated patients",
                f"{n_patients} each · FOR2107's correlations · {splits}×",
            )

    # g: reading the result
    axis = figure.add_subplot(bottom[1])
    _clean(axis, "y")
    grid = np.linspace(0, 1, 200)
    for source, centre in (("null", 0.3), ("planted", 0.7)):
        density = np.exp(-0.5 * ((grid - centre) / 0.08) ** 2)
        axis.fill_between(grid, density, color=STYLE[source]["color"], alpha=0.3, linewidth=0)
        axis.plot(grid, density, color=STYLE[source]["color"], linewidth=1.4)
        axis.text(
            centre,
            0.45,
            SHORT[source].replace(" ", "\n", 1),
            ha="center",
            va="center",
            fontsize=8,
            color=STYLE[source]["color"] if source != "null" else TEXT_MUTED,
            fontweight="bold",
        )
    for x, text in ((0.3, "FOR2107 here:\nno subgroups"), (0.7, "FOR2107 here:\nTozzi replicates")):
        axis.annotate(
            text,
            (x, 1.04),
            xytext=(x, 1.35),
            ha="center",
            va="bottom",
            fontsize=8,
            color=STYLE["observed"]["color"],
            arrowprops={"arrowstyle": "-|>", "color": STYLE["observed"]["color"], "linewidth": 1.2},
        )
    axis.set_ylim(0, 1.75)
    axis.set_yticks([])
    axis.set_xticks([0, 1], ["low", "high"])
    axis.set_xlabel("reproducibility (separation or ARI)")
    _panel_label(
        axis, "g", "Where does FOR2107 fall?", "if gray and orange overlap, the test cannot decide"
    )

    # h: further checks
    axis = _schematic(figure.add_subplot(bottom[2]))
    _panel_label(axis, "h", "Two further checks", "on all patients, without splitting")
    axis.text(
        0.0,
        0.93,
        "Same patients measured twice: same groups?",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT,
        va="center",
    )
    for index, scan in enumerate(("AP task", "FM task", "rest scan")):
        _box(axis, 0.02 + index * 0.33, 0.66, 0.26, 0.15, f"{scan}\n14 connections", size=8)
    _arrow(axis, (0.15, 0.66), (0.42, 0.53))
    _arrow(axis, (0.48, 0.66), (0.48, 0.55))
    _arrow(axis, (0.81, 0.66), (0.56, 0.53))
    _box(axis, 0.18, 0.42, 0.64, 0.11, "cluster each into 6 · compare with ARI", size=8)
    axis.text(
        0.0,
        0.3,
        "Do the clusters look like Tozzi's six biotypes?",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT,
        va="center",
    )
    _box(axis, 0.02, 0.04, 0.4, 0.17, "FOR2107 cluster profiles\non his 31 features", size=8)
    _box(axis, 0.58, 0.04, 0.4, 0.17, "Tozzi's published\nbiotype profiles", size=8)
    axis.annotate(
        "",
        (0.58, 0.125),
        (0.42, 0.125),
        arrowprops={"arrowstyle": "<|-|>", "color": TEXT_MUTED, "linewidth": 1.4},
    )
    axis.text(0.5, 0.155, "cosine", ha="center", fontsize=8, color=TEXT_MUTED)

    # reading order of the first row
    figure.canvas.draw()
    panels = [ax for ax in figure.axes][:6]
    row_y = (panels[0].get_position().y0 + panels[0].get_position().y1) / 2
    for left, right in ((0, 1), (1, 2), (3, 4), (4, 5)):
        a, b = panels[left].get_position(), panels[right].get_position()
        # stop short of the next panel's axis labels
        end = b.x0 - (0.03 if right == 4 else 0.004)
        figure.patches.append(
            plt.matplotlib.patches.FancyArrowPatch(
                (a.x1 + 0.002, row_y),
                (end, row_y),
                transform=figure.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                color=TEXT_MUTED,
                linewidth=1.4,
            )
        )

    caption = figure.add_subplot(outer[2])
    caption.axis("off")
    caption.text(
        0,
        1,
        "\n".join(textwrap.wrap(METHOD_CAPTION, 195)),
        va="top",
        ha="left",
        fontsize=8.5,
        color=TEXT,
        linespacing=1.4,
    )
    figure.savefig(path, bbox_inches="tight", pad_inches=0.25, dpi=200)
    plt.close(figure)
    return path


METHOD_CAPTION = (
    "Figure 1 | How reproducible are six patient subgroups? Schematic with toy data. a, Every patient has a "
    "profile of brain-circuit features, each expressed relative to healthy controls. The analysis is repeated "
    "for 106 feature sets: Tozzi's 31 construct-matched features, all 285 features of his networks, sets with "
    "one network, task or feature type left out, and random subsets. b, The 701 patients are split at random "
    "into two halves, 100 times. c, Each half is clustered into six groups as in Tozzi et al. (2024). d, Each "
    "half-A cluster is paired with the half-B cluster whose average profile is most similar (cosine). Real "
    "subgroups come back: high similarity on the diagonal (matched pairs), low similarity elsewhere "
    "(unmatched). e, Each half-B patient is also assigned to the closest half-A profile; the adjusted Rand "
    "index (ARI) measures how often this agrees with half B's own clustering. f, Steps b–e are repeated on "
    "simulated patients without subgroups (gray) and on simulated patients carrying Tozzi's six biotypes at "
    "the strength calibrated to his study (orange). g, FOR2107 is judged by where it falls between these two "
    "references. h, Two further checks: whether the same connections measured in different scans group the "
    "same patients, and whether the cluster profiles resemble Tozzi's published biotypes."
)
