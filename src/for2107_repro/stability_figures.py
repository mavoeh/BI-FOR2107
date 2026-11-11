"""Figures for the split-half stability analysis, drawn from its saved tables.

Each figure sets the observed data (blue) against the two simulated
references: no clusters (gray) and Tozzi's biotypes planted at calibrated
strength (orange). Marker shapes repeat the distinction for print and
colour-vision deficiency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e6e5e0"
BAND = "#f0efec"
STYLE = {
    "observed": {"color": "#2a78d6", "marker": "o", "label": "FOR2107 (observed)"},
    "null": {"color": "#8a8984", "marker": "s", "label": "No subgroups (simulated)"},
    "planted": {"color": "#eb6834", "marker": "D", "label": "Tozzi's biotypes (simulated)"},
}
ORDER = ("null", "planted", "observed")
COSINE = LinearSegmentedColormap.from_list(
    "cosine", ["#9e2b2b", "#e34948", BAND, "#2a78d6", "#104281"]
)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 10,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "svg.fonttype": "none",
    }
)


def _label(name: str, n_features: int | None = None) -> str:
    """Readable feature-set name."""
    text = name
    if name == "tozzi_matched":
        text = "Tozzi's 31 matched features"
    elif name == "tozzi_task_free":
        text = "Tozzi's 14 task-free edges"
    elif name.startswith("tozzi_without_"):
        text = f"Tozzi's 31 without {name.removeprefix('tozzi_without_')}"
    elif name == "all":
        text = "All features"
    elif name == "extensions":
        text = "Extensions only"
    elif name == "task_free_all":
        text = "All task-free edges"
    elif name.startswith("without_"):
        part = name.removeprefix("without_")
        part = part.replace("MatchFaces_vs_MatchShapes", "faces>shapes").replace("_vs_", ">")
        text = f"All without {part.replace('_', ' ')}"
    return f"{text}  ({n_features})" if n_features is not None else text


def _axis(axis: plt.Axes) -> None:
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.tick_params(axis="y", length=0)


def _legend(figure: plt.Figure, extra: list[Line2D] | None = None) -> None:
    handles = [
        Line2D(
            [],
            [],
            color=STYLE[s]["color"],
            marker=STYLE[s]["marker"],
            linestyle="",
            markersize=7,
            label=STYLE[s]["label"],
        )
        for s in ("observed", "null", "planted")
    ]
    figure.legend(
        handles=handles + (extra or []),
        loc="upper center",
        ncol=len(handles) + len(extra or []),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )


def _structured_order(summary: pd.DataFrame) -> list[str]:
    structured = summary.loc[summary.kind == "structured"]
    names = list(dict.fromkeys(structured.feature_set))
    tozzi = [
        "tozzi_matched",
        *sorted(n for n in names if n.startswith("tozzi_without_")),
        "tozzi_task_free",
    ]
    broad = [
        "all",
        "extensions",
        "task_free_all",
        *sorted(n for n in names if n.startswith("without_")),
    ]
    return [n for n in tozzi + broad if n in names]


def split_half_figure(summary: pd.DataFrame, path: Path, min_auc: float = 0.75) -> Path:
    """Median (IQR) separation and assignment ARI per structured feature set."""
    long = summary.set_index(["feature_set", "source"])
    wide = summary.drop_duplicates("feature_set").set_index("feature_set")
    names = _structured_order(summary)
    figure, axes = plt.subplots(
        1, 2, figsize=(11, 0.32 * len(names) + 1.6), sharey=True, layout="constrained"
    )
    y = np.arange(len(names))
    offsets = {"null": -0.18, "planted": 0.18, "observed": 0.0}
    metrics = (
        ("separation", "Centroid separation\n(matched − unmatched cosine)"),
        ("ari_assignment", "Assignment agreement between halves\n(ARI, Tozzi's split-half rule)"),
    )
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        _axis(axis)
        # rows where a single split tells planted biotypes from no clusters
        for row, name in enumerate(names):
            if wide.loc[name, f"{metric}_auc"] >= min_auc:
                axis.axhspan(row - 0.45, row + 0.45, color=BAND, zorder=0, linewidth=0)
        for source in ORDER:
            style = STYLE[source]
            median = [long.loc[(n, source), f"{metric}_median"] for n in names]
            low = [long.loc[(n, source), f"{metric}_q25"] for n in names]
            high = [long.loc[(n, source), f"{metric}_q75"] for n in names]
            axis.hlines(
                y + offsets[source], low, high, color=style["color"], linewidth=1.5, zorder=2
            )
            axis.plot(
                median,
                y + offsets[source],
                linestyle="",
                marker=style["marker"],
                markersize=6,
                color=style["color"],
                markeredgecolor=SURFACE,
                markeredgewidth=1,
                zorder=3,
            )
        axis.set_title(title, loc="left")
        separator = names.index("all") - 0.5
        axis.axhline(separator, color=TEXT_MUTED, linewidth=0.6)
    sizes = wide.loc[names, "n_features"].astype(int)
    axes[0].set_yticks(y, [_label(n, s) for n, s in zip(names, sizes, strict=True)])
    axes[0].invert_yaxis()
    band = Line2D(
        [],
        [],
        color=BAND,
        marker="s",
        linestyle="",
        markersize=10,
        label=f"planted vs null AUC ≥ {min_auc}",
    )
    _legend(figure, [band])
    figure.get_layout_engine().set(rect=(0, 0, 1, 0.94))
    figure.savefig(path)
    plt.close(figure)
    return path


def matrices_figure(matrices: pd.DataFrame, sets: list[str], path: Path) -> Path:
    """Mean matched 6 x 6 cosine matrices (rows: half A by size)."""
    figure, axes = plt.subplots(len(sets), 3, figsize=(9, 3.0 * len(sets)), layout="constrained")
    axes = np.atleast_2d(axes)
    image = None
    for row, name in enumerate(sets):
        for column, source in enumerate(("null", "observed", "planted")):
            axis = axes[row, column]
            cells = matrices.loc[(matrices.feature_set == name) & (matrices.source == source)]
            k = int(cells.row_rank.max())
            values = np.full((k, k), np.nan)
            values[cells.row_rank - 1, cells.column_rank - 1] = cells.cosine
            image = axis.imshow(values, vmin=-1, vmax=1, cmap=COSINE)
            for index in range(k):
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
            ticks = range(k)
            axis.set_xticks(ticks, [t + 1 for t in ticks])
            axis.set_yticks(ticks, [t + 1 for t in ticks])
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_title(
                f"{STYLE[source]['label']}",
                fontsize=9,
                color=STYLE[source]["color"] if source != "null" else TEXT_MUTED,
            )
            if column == 0:
                axis.set_ylabel(f"{_label(name)}\nhalf A cluster (largest first)")
            if row == len(sets) - 1:
                axis.set_xlabel("matched half-B cluster")
    figure.colorbar(image, ax=axes, shrink=0.6, label="cosine of cluster centroids")
    figure.savefig(path)
    plt.close(figure)
    return path


def random_sets_figure(summary: pd.DataFrame, path: Path) -> Path:
    """Random leave-out subsets: medians per draw and their trend."""
    random = summary.loc[summary.kind == "random"]
    pools = [p for p in ("tozzi_matched", "all") if p in set(random.pool)]
    metrics = (("separation", "Centroid separation"), ("ari_assignment", "Assignment ARI"))
    figure, axes = plt.subplots(
        len(metrics),
        len(pools),
        figsize=(4.6 * len(pools), 6.4),
        layout="constrained",
        squeeze=False,
    )
    rng = np.random.default_rng(0)
    for column, pool in enumerate(pools):
        subset = random.loc[random.pool == pool]
        fractions = sorted(subset.leave_out.unique())
        size_of = subset.groupby("leave_out").n_features.median()
        for row, (metric, title) in enumerate(metrics):
            axis = axes[row, column]
            axis.grid(axis="y", color=GRID, linewidth=0.6)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            for source in ORDER:
                style = STYLE[source]
                values = subset.loc[subset.source == source]
                x = values.leave_out.map({f: i for i, f in enumerate(fractions)}).to_numpy(float)
                axis.plot(
                    x + rng.uniform(-0.12, 0.12, len(x)),
                    values[f"{metric}_median"],
                    linestyle="",
                    marker=style["marker"],
                    markersize=3.5,
                    alpha=0.45,
                    color=style["color"],
                )
                trend = values.groupby("leave_out")[f"{metric}_median"].median().reindex(fractions)
                axis.plot(
                    range(len(fractions)),
                    trend,
                    color=style["color"],
                    linewidth=2,
                    marker=style["marker"],
                    markersize=6,
                    markeredgecolor=SURFACE,
                )
            axis.set_xticks(
                range(len(fractions)),
                [f"{round(100 * f)}%\n({int(size_of[f])})" for f in fractions],
            )
            if row == len(metrics) - 1:
                axis.set_xlabel("features left out at random (features kept)")
            if column == 0:
                axis.set_ylabel(title)
            if row == 0:
                axis.set_title(
                    "Pool: Tozzi's 31 matched features"
                    if pool == "tozzi_matched"
                    else "Pool: all 285 features",
                    loc="left",
                )
    _legend(figure)
    figure.get_layout_engine().set(rect=(0, 0, 1, 0.93))
    figure.savefig(path)
    plt.close(figure)
    return path


def _draw_rows(axis: plt.Axes, draws: pd.DataFrame, rows: list[str], key: str, metric: str) -> None:
    rng = np.random.default_rng(1)
    _axis(axis)
    for row, name in enumerate(rows):
        group = draws.loc[draws[key] == name]
        for source, offset in (("null", -0.17), ("planted", 0.17)):
            values = group.loc[group.source == source, metric].dropna()
            style = STYLE[source]
            axis.plot(
                values,
                row + offset + rng.uniform(-0.08, 0.08, len(values)),
                linestyle="",
                marker=style["marker"],
                markersize=2.5,
                alpha=0.35,
                color=style["color"],
            )
            axis.plot(
                [values.median()],
                [row + offset],
                linestyle="",
                marker="|",
                markersize=12,
                markeredgewidth=2,
                color=style["color"],
            )
        observed = group.loc[group.source == "observed", metric]
        axis.plot(
            observed,
            [row] * len(observed),
            linestyle="",
            marker="o",
            markersize=8,
            color=STYLE["observed"]["color"],
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=4,
        )
    axis.set_yticks(range(len(rows)))
    axis.invert_yaxis()


def cross_set_figure(draws: pd.DataFrame, path: Path) -> Path:
    """Agreement between clusterings of the same patients: observed vs references."""
    independent = list(
        dict.fromkeys(draws.loc[draws.kind == "independent_measurement", "comparison"])
    )
    structured = [
        name
        for name in dict.fromkeys(draws.loc[draws.kind == "same_measurement", "comparison"])
        if not name.startswith("random_")
    ]
    panels = [
        (independent, "Same edges, independent measurements"),
        (structured, "Two feature sets, same data"),
    ]
    panels = [(rows, title) for rows, title in panels if rows]
    heights = [len(rows) for rows, _ in panels]
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9, 0.42 * sum(heights) + 2.2),
        layout="constrained",
        gridspec_kw={"height_ratios": heights},
        squeeze=False,
    )
    names = {
        "edges_apfm": "AP + FM residuals",
        "edges_ap": "AP run",
        "edges_fm": "FM run",
        "edges_rest": "rest scan",
    }
    for axis, (rows, title) in zip(axes[:, 0], panels, strict=True):
        _draw_rows(axis, draws, rows, "comparison", "ari")
        labels = []
        for name in rows:
            label = name
            for key, value in names.items():
                label = label.replace(key, value)
            label = label.replace("(all_edges)", "· all 51 edges").replace(
                "(tozzi_edges)", "· Tozzi's 14"
            )
            if " | " in label:
                label = " vs ".join(_label(part) for part in label.split(" | "))
            labels.append(label)
        axis.set_yticklabels(labels)
        axis.set_title(title, loc="left")
        axis.axvline(0, color=TEXT_MUTED, linewidth=0.6)
    axes[-1, 0].set_xlabel("Agreement of the two six-cluster solutions (ARI)")
    _legend(figure)
    figure.get_layout_engine().set(rect=(0, 0, 1, 1 - 0.5 / (0.42 * sum(heights) + 2.2)))
    figure.savefig(path)
    plt.close(figure)
    return path


def tozzi_match_figure(draws: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    """Cosine of FOR2107 cluster profiles with Tozzi's six biotypes."""
    structured = set(summary.loc[summary.kind == "structured", "feature_set"])
    rows = [
        name for name in _structured_order(summary) if name in set(draws.feature_set) & structured
    ]
    figure, axis = plt.subplots(figsize=(9, 0.38 * len(rows) + 1.8), layout="constrained")
    _draw_rows(axis, draws, rows, "feature_set", "mean_matched")
    sizes = summary.drop_duplicates("feature_set").set_index("feature_set").n_features
    axis.set_yticklabels([_label(name, int(sizes[name])) for name in rows])
    axis.set_xlabel(
        "Mean cosine of the six matched cluster profiles with Tozzi's biotypes\n(circuit scores of his features)"
    )
    _legend(figure)
    figure.get_layout_engine().set(rect=(0, 0, 1, 1 - 0.45 / (0.38 * len(rows) + 1.8)))
    figure.savefig(path)
    plt.close(figure)
    return path


def read_table(path: Path) -> pd.DataFrame:
    """Read a stability table; the source label "null" is text, not missing."""
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=["", "nan", "NaN"])


def write_stability_figures(root: Path) -> list[str]:
    """Draw every figure whose input tables exist under `root`."""
    root = Path(root)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    written: list[Path] = []
    summary_path = root / "split_half_summary.tsv"
    summary = read_table(summary_path) if summary_path.exists() else None
    if summary is not None:
        written.append(split_half_figure(summary, out / "split_half_structured.svg"))
        written.append(random_sets_figure(summary, out / "split_half_random_sets.svg"))
    matrices_path = root / "mean_cosine_matrices.tsv"
    if matrices_path.exists():
        matrices = read_table(matrices_path)
        sets = [s for s in ("tozzi_matched", "all") if s in set(matrices.feature_set)]
        written.append(matrices_figure(matrices, sets, out / "split_half_matrices.svg"))
    cross_path = root / "cross_set_draws.tsv"
    if cross_path.exists():
        written.append(cross_set_figure(read_table(cross_path), out / "cross_set_agreement.svg"))
    tozzi_path = root / "tozzi_match_draws.tsv"
    if tozzi_path.exists() and summary is not None:
        written.append(
            tozzi_match_figure(read_table(tozzi_path), summary, out / "tozzi_profile_match.svg")
        )
    written.extend(_overview_figures(root, out, summary))
    return [str(path) for path in written]


def _overview_figures(root: Path, out: Path, summary: pd.DataFrame | None) -> list[Path]:
    """Figure 1 (method schematic) and, once every table exists, Figure 2 (results)."""
    import json

    from .stability_overview import (
        METHOD_CAPTION,
        RESULTS_CAPTION,
        method_overview,
        results_overview,
    )

    status_path = root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    counts = {
        key: int(status[name])
        for key, name in (
            ("n_patients", "patients"),
            ("n_features", "features"),
            ("n_sets", "feature_sets"),
            ("splits", "splits"),
        )
        if name in status
    }
    written = []
    captions = [METHOD_CAPTION]
    for suffix in (".svg", ".png"):
        written.append(method_overview(out / f"figure1_method{suffix}", **counts))
    tables = (
        "split_half_splits.tsv",
        "mean_cosine_matrices.tsv",
        "cross_set_draws.tsv",
        "tozzi_match_draws.tsv",
    )
    if summary is not None and all((root / name).exists() for name in tables):
        for suffix in (".svg", ".png"):
            written.append(results_overview(root, out / f"figure2_results{suffix}"))
        captions.append(RESULTS_CAPTION)
    (out / "captions.md").write_text("\n\n".join(captions) + "\n", encoding="utf-8")
    return written
