from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, PathPatch
from matplotlib.path import Path
from scipy import stats

GROUP_COLORS = {
    "default": "#7096cf",
    "salience": "#3bab68",
    "attention": "#edcf3f",
}

EDGE_SPEC = {
    "D2D1": ("D2", "D1", "default"),
    "D1D3": ("D1", "D3", "default"),
    "D1D4": ("D1", "D4", "default"),
    "D2D4": ("D2", "D4", "default"),
    "D3D4": ("D3", "D4", "default"),
    "S1S3": ("S1", "S3", "salience"),
    "S2S4": ("S2", "S4", "salience"),
    "S1S2": ("S1", "S2", "salience"),
    "A2A1": ("A2", "A1", "attention"),
    "A3A1": ("A3", "A1", "attention"),
    "A4A2": ("A4", "A2", "attention"),
    "A5A3": ("A5", "A3", "attention"),
    "A4A6": ("A4", "A6", "attention"),
    "A5A7": ("A5", "A7", "attention"),
}

NODE_MNI_XY = {
    # Centers of mass measured directly from the published Tozzi ROI masks.
    "D1": (-1.8, 49.9),
    "D2": (-45.5, -69.5),
    "D3": (50.2, -61.7),
    "D4": (0.0, -50.2),
    "S1": (-37.8, 15.4),
    "S2": (36.9, 17.5),
    "S3": (-23.4, -3.3),
    "S4": (24.8, -6.0),
    "A1": (-1.2, 13.0),
    "A2": (-43.7, 5.4),
    "A3": (49.8, 9.5),
    "A4": (-31.5, -54.3),
    "A5": (37.1, -55.4),
    "A6": (-15.1, -66.6),
    "A7": (17.5, -67.0),
}

# Orthographic dorsal projection: MNI x is left/right and MNI y is
# posterior/anterior. Fixed anatomical scaling preserves relative locations.
NODE_POSITIONS = {
    node: (mni_x / 75.0, mni_y / 105.0) for node, (mni_x, mni_y) in NODE_MNI_XY.items()
}


def _feature_code(name: str) -> str:
    return name.split("_", maxsplit=1)[0]


def _draw_brain_outline(ax: plt.Axes) -> None:
    outline = Ellipse(
        (0, 0),
        width=1.90,
        height=1.85,
        facecolor="#fbfbfa",
        edgecolor="#59616a",
        linewidth=0.9,
        zorder=0,
    )
    ax.add_patch(outline)
    ax.plot([0, 0], [-0.78, 0.78], color="#a5abb0", linewidth=0.55, zorder=0)
    curves = [
        [(-0.72, 0.44), (-0.42, 0.58), (-0.18, 0.40)],
        [(0.72, 0.44), (0.42, 0.58), (0.18, 0.40)],
        [(-0.78, 0.05), (-0.45, 0.18), (-0.20, 0.02)],
        [(0.78, 0.05), (0.45, 0.18), (0.20, 0.02)],
        [(-0.70, -0.34), (-0.42, -0.20), (-0.18, -0.36)],
        [(0.70, -0.34), (0.42, -0.20), (0.18, -0.36)],
        [(-0.48, -0.67), (-0.30, -0.49), (-0.10, -0.62)],
        [(0.48, -0.67), (0.30, -0.49), (0.10, -0.62)],
    ]
    for start, control, end in curves:
        path = Path([start, control, end], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        ax.add_patch(PathPatch(path, fill=False, edgecolor="#c4c8cb", linewidth=0.45, zorder=0))


def _draw_intrinsic_schematic(
    ax: plt.Axes,
    profile: Mapping[str, float],
    significance: Mapping[str, bool],
    *,
    threshold: float,
) -> None:
    _draw_brain_outline(ax)
    active_nodes: set[str] = set()
    node_groups: dict[str, str] = {}
    for feature, value in profile.items():
        code = _feature_code(feature)
        if code not in EDGE_SPEC or not np.isfinite(value) or abs(value) < threshold:
            continue
        first, second, group = EDGE_SPEC[code]
        active_nodes.update((first, second))
        is_significant = bool(significance.get(feature, False))
        if is_significant:
            node_groups[first] = group
            node_groups[second] = group
        x = [NODE_POSITIONS[first][0], NODE_POSITIONS[second][0]]
        y = [NODE_POSITIONS[first][1], NODE_POSITIONS[second][1]]
        ax.plot(
            x,
            y,
            color=GROUP_COLORS[group] if is_significant else "#b8bdc0",
            linewidth=min(5.2, 1.4 + 1.8 * abs(value)),
            linestyle="-" if value > 0 else (0, (2.2, 1.5)),
            alpha=0.95,
            solid_capstyle="round",
            zorder=2,
        )
    for node, (x, y) in NODE_POSITIONS.items():
        group = node_groups.get(node)
        ax.scatter(
            x,
            y,
            s=115 if node in active_nodes else 48,
            color=GROUP_COLORS[group] if group else "#d4d7d9",
            edgecolor="#6c7379" if group else "#aeb3b6",
            linewidth=0.45,
            zorder=3,
        )
    ax.text(0, 0.94, "A", ha="center", va="bottom", fontsize=6.5, color="#8b9195")
    ax.text(0, -0.94, "P", ha="center", va="top", fontsize=6.5, color="#8b9195")
    ax.set(xlim=(-1.04, 1.04), ylim=(-1.00, 1.00), aspect="equal")
    ax.axis("off")


def cluster_deviation_tests(
    assignments: pd.DataFrame,
    feature_names: list[str] | None = None,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test cluster-edge means against the healthy-reference value of zero.

    These tests are descriptive because the same features defined the clusters.
    FDR correction is applied jointly across every cluster-by-edge comparison.
    """
    if feature_names is None:
        feature_names = [
            name
            for name in assignments.columns
            if name not in {"subject_id", "cluster"} and _feature_code(name) in EDGE_SPEC
        ]
    rows = []
    for cluster, selected in assignments.groupby("cluster", sort=True):
        for feature in feature_names:
            values = pd.to_numeric(selected[feature], errors="coerce").dropna().to_numpy()
            statistic, p_value = (
                stats.ttest_1samp(values, popmean=0.0) if len(values) >= 2 else (np.nan, np.nan)
            )
            mean = float(np.mean(values)) if len(values) else np.nan
            sem = (
                float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) >= 2 else np.nan
            )
            ci_half_width = (
                float(stats.t.ppf(0.975, len(values) - 1) * sem) if len(values) >= 2 else np.nan
            )
            rows.append(
                {
                    "cluster": int(cluster),
                    "feature": feature,
                    "n": len(values),
                    "mean": mean,
                    "sem": sem,
                    "ci_95_low": mean - ci_half_width,
                    "ci_95_high": mean + ci_half_width,
                    "ci_95_half_width": ci_half_width,
                    "t_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
    result = pd.DataFrame(rows)
    valid = result["p_value"].notna()
    result["p_fdr"] = np.nan
    if valid.any():
        result.loc[valid, "p_fdr"] = stats.false_discovery_control(
            result.loc[valid, "p_value"].to_numpy(), method="bh"
        )
    result["significant_fdr"] = result["p_fdr"] < alpha
    result["test"] = "two-sided one-sample t-test versus healthy-reference zero"
    result["multiplicity"] = "Benjamini-Hochberg across all cluster-edge tests"
    result["post_clustering_descriptive"] = True
    return result


def tozzi_style_rest_figure(
    assignments: pd.DataFrame,
    *,
    threshold: float = 0.5,
    deviation_tests: pd.DataFrame | None = None,
) -> plt.Figure:
    """Recreate the layout of Tozzi Fig. 3 using rest-only cluster profiles."""
    feature_names = [
        name
        for name in assignments.columns
        if name not in {"subject_id", "cluster"} and _feature_code(name) in EDGE_SPEC
    ]
    if not feature_names:
        raise ValueError("No Tozzi task-free edge features found")
    clusters = sorted(assignments["cluster"].dropna().astype(int).unique())
    if not clusters:
        raise ValueError("No cluster assignments found")
    if deviation_tests is None:
        deviation_tests = cluster_deviation_tests(assignments, feature_names)
    significant = {
        (int(row.cluster), str(row.feature)): bool(row.significant_fdr)
        for row in deviation_tests.itertuples()
    }

    figure = plt.figure(figsize=(12.8, 2.05 * len(clusters) + 1.5), constrained_layout=False)
    grid = figure.add_gridspec(
        len(clusters),
        2,
        width_ratios=(1.25, 6.1),
        hspace=0.10,
        wspace=0.08,
        left=0.045,
        right=0.875,
        top=0.935,
        bottom=0.12,
    )
    bar_axes: list[plt.Axes] = []
    x = np.arange(len(feature_names))
    feature_colors = [GROUP_COLORS[EDGE_SPEC[_feature_code(name)][2]] for name in feature_names]
    rng = np.random.default_rng(2107)

    finite = assignments[feature_names].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    extent = float(np.quantile(np.abs(finite), 0.995)) if len(finite) else 5.0
    y_limit = max(5.0, min(8.0, float(np.ceil(extent))))

    for row, cluster in enumerate(clusters):
        selected = assignments.loc[assignments["cluster"].astype(int) == cluster]
        means = selected[feature_names].mean()
        confidence_errors = (
            deviation_tests.loc[deviation_tests["cluster"].astype(int) == cluster]
            .set_index("feature")
            .reindex(feature_names)["ci_95_half_width"]
            .to_numpy(dtype=float)
        )

        brain_ax = figure.add_subplot(grid[row, 0])
        cluster_significance = {
            feature: significant.get((cluster, feature), False) for feature in feature_names
        }
        _draw_intrinsic_schematic(
            brain_ax,
            means.to_dict(),
            cluster_significance,
            threshold=threshold,
        )
        brain_ax.set_title(
            f"Cluster {cluster}\n(n = {len(selected)})",
            fontsize=10.5,
            fontweight="semibold",
            pad=0,
        )

        ax = figure.add_subplot(grid[row, 1], sharex=bar_axes[0] if bar_axes else None)
        bar_axes.append(ax)
        for feature_index, feature in enumerate(feature_names):
            values = pd.to_numeric(selected[feature], errors="coerce").dropna().to_numpy()
            jitter = rng.uniform(-0.30, 0.30, size=len(values))
            ax.scatter(
                np.full(len(values), feature_index) + jitter,
                values,
                s=5,
                color="#72777c",
                alpha=0.16,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
        bar_colors = [
            color if significant.get((cluster, feature), False) else "#c8cccf"
            for feature, color in zip(feature_names, feature_colors, strict=True)
        ]
        ax.bar(
            x,
            means.to_numpy(dtype=float),
            yerr=confidence_errors,
            width=0.66,
            color=bar_colors,
            edgecolor="none",
            error_kw={"ecolor": "#23272a", "elinewidth": 0.8, "capsize": 2},
            zorder=3,
        )
        ax.axhline(0, color="#34383b", linewidth=0.65, zorder=2)
        ax.axhline(threshold, color="#b5b8ba", linewidth=0.55, linestyle=(0, (2, 2)))
        ax.axhline(-threshold, color="#b5b8ba", linewidth=0.55, linestyle=(0, (2, 2)))
        ax.axvline(4.5, color="#d7d9da", linewidth=0.65)
        ax.axvline(7.5, color="#d7d9da", linewidth=0.65)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_yticks([-5, -2.5, 0, 2.5, 5] if y_limit == 5 else [-y_limit, 0, y_limit])
        ax.tick_params(axis="y", labelsize=8, length=2, colors="#4e555a")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#858b8f")
        ax.spines[["left", "bottom"]].set_linewidth(0.7)
        if row < len(clusters) - 1:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            ax.set_xticks(
                x, [_feature_code(name) for name in feature_names], rotation=58, ha="right"
            )
            ax.tick_params(axis="x", labelsize=8, pad=1)

    figure.text(
        0.207,
        0.53,
        "Circuit score (healthy-reference SD)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
    )
    figure.suptitle(
        "FOR2107 task-free connection profiles",
        x=0.46,
        y=0.985,
        fontsize=15,
        fontweight="semibold",
    )
    legend_handles = [
        Patch(facecolor=GROUP_COLORS["default"], label="Default mode"),
        Patch(facecolor=GROUP_COLORS["attention"], label="Attention"),
        Patch(facecolor=GROUP_COLORS["salience"], label="Salience"),
        Patch(facecolor="#c8cccf", label="Not FDR-significant"),
        Line2D([0], [0], color="#4f5559", linewidth=2.5, label=f"> +{threshold:g} SD"),
        Line2D(
            [0],
            [0],
            color="#4f5559",
            linewidth=2.5,
            linestyle=(0, (2.2, 1.5)),
            label=f"< -{threshold:g} SD",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.885, 0.82),
        frameon=False,
        title="Intrinsic circuits",
        fontsize=9,
        title_fontsize=10,
    )
    return figure
