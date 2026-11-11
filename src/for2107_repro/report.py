from __future__ import annotations

import base64
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, select_autoescape
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import t
from sklearn.metrics import silhouette_samples

from .clustering import empirical_p_value, fit_clusters
from .clustering_figures import supplementary_clustering_figures
from .config import Config
from .tozzi_style import cluster_deviation_tests, tozzi_style_rest_figure

COLORS = {
    "blue": "#176b87",
    "teal": "#19a7a0",
    "amber": "#d99022",
    "slate": "#64748b",
    "ink": "#172033",
    "line": "#cbd5e1",
}
SERIES_COLORS = [
    COLORS["blue"],
    COLORS["amber"],
    COLORS["teal"],
    "#7c6eb0",
    "#dc6b63",
    "#4f8f5b",
]

EDGE_LABELS = {
    "D2D1": "D2–D1\namPFC–left AG",
    "D1D3": "D1–D3\namPFC–right AG",
    "D1D4": "D1–D4\namPFC–PCC",
    "D2D4": "D2–D4\nleft AG–PCC",
    "D3D4": "D3–D4\nright AG–PCC",
    "S1S3": "S1–S3\nleft AI–amygdala",
    "S2S4": "S2–S4\nright AI–amygdala",
    "S1S2": "S1–S2\nleft–right AI",
    "A2A1": "A2–A1\nleft LPFC–msPFC",
    "A3A1": "A3–A1\nright LPFC–msPFC",
    "A4A2": "A4–A2\nleft aIPL–LPFC",
    "A5A3": "A5–A3\nright aIPL–LPFC",
    "A4A6": "A4–A6\nleft aIPL–precuneus",
    "A5A7": "A5–A7\nright aIPL–precuneus",
}

REPORT_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
@page { size: A4; margin: 17mm 15mm 18mm; @bottom-right { content: counter(page); color: #64748b; } }
:root { --ink:#172033; --muted:#64748b; --line:#dbe4ee; --blue:#176b87; --teal:#19a7a0; --amber:#d99022; --paper:#fff; --wash:#f3f7fa; }
* { box-sizing:border-box } body { margin:0; font:10.5pt/1.48 Inter, "DejaVu Sans", sans-serif; color:var(--ink); background:var(--paper) }
header { border-bottom:3px solid var(--teal); padding:0 0 12px; margin-bottom:22px } h1 { font-size:25pt; line-height:1.12; margin:0 0 7px; letter-spacing:-.5px }
h2 { font-size:15pt; margin:24px 0 9px; color:var(--blue); break-after:avoid } h3 { font-size:11.5pt; margin:16px 0 5px }
p { margin:6px 0 10px } .subtitle { color:var(--muted); font-size:11pt }.badge { display:inline-block; border-radius:999px; padding:4px 10px; color:white; background:var(--amber); font-weight:700; letter-spacing:.25px }
.grid { display:grid; grid-template-columns:repeat(4,1fr); gap:9px; margin:14px 0 }.card { background:var(--wash); border:1px solid var(--line); border-radius:8px; padding:11px }.metric { font-size:19pt; font-weight:750; color:var(--blue) }.label { color:var(--muted); font-size:8.5pt; text-transform:uppercase; letter-spacing:.5px }
.callout { border-left:4px solid var(--amber); background:#fff8eb; padding:10px 13px; margin:13px 0 }.ok { border-left-color:var(--teal); background:#edfafa }.figure { break-inside:avoid; margin:15px 0 18px }.figure img { width:100%; max-height:145mm; object-fit:contain }.caption { color:var(--muted); font-size:8.5pt; margin-top:4px }
table { width:100%; border-collapse:collapse; font-size:8.2pt; margin:8px 0 15px } th { text-align:left; background:#eaf1f6; color:#24445a } th,td { border-bottom:1px solid var(--line); padding:5px 6px; vertical-align:top } code { font-family:"DejaVu Sans Mono",monospace; font-size:8.5pt } footer { margin-top:25px; border-top:1px solid var(--line); padding-top:8px; color:var(--muted); font-size:8pt }
pre { white-space:pre-wrap; overflow-wrap:anywhere; font-size:6.8pt; line-height:1.25; break-inside:auto }
</style></head><body>
<header><span class="badge">{{ mode|upper }} ANALYSIS</span><h1>{{ title }}</h1><div class="subtitle">{{ subtitle }} · generated {{ created }}</div></header>
<section class="grid"><div class="card"><div class="metric">{{ summary.subject_count }}</div><div class="label">Subjects inventoried</div></div><div class="card"><div class="metric">{{ summary.cohort.clinical_with_imaging }}</div><div class="label">Selected clinical</div></div><div class="card"><div class="metric">{{ summary.cohort.healthy_reference_with_imaging }}</div><div class="label">Healthy reference</div></div><div class="card"><div class="metric">{{ summary.missing_task_events }}</div><div class="label">Task scans missing events</div></div></section>
{% if development %}<div class="callout"><strong>Interpretation boundary.</strong> This report exercises the analysis machinery on a development dataset. It does not establish a cluster count, biotype, or clinical association.</div>{% endif %}
<h2>Executive status</h2><p>{{ status }}</p>
{% if limitations %}<h3>Current limitations</h3><ul>{% for item in limitations %}<li>{{ item }}</li>{% endfor %}</ul>{% endif %}
{% for figure in figures %}<div class="figure"><img src="{{ figure.uri }}"><div class="caption">{{ figure.caption }}</div></div>{% endfor %}
{% if smoke_table %}<h2>Resting-state extraction result</h2><p>This compact summary confirms that the preprocessed resting-state series can be transformed into the configured circuit features. The deidentified participant-level values remain available in the analysis outputs for auditing.</p>
{% if smoke_figure %}<div class="figure"><img src="{{ smoke_figure.uri }}"><div class="caption">{{ smoke_figure.caption }}</div></div>{% endif %}
{{ smoke_table|safe }}{% endif %}
<h2>Feature specification</h2><p>The primary feature tier uses resting-state, face matching, and affective priming. Encoding remains exploratory and is not treated as a substitute for Tozzi's Go-NoGo cognitive-control construct.</p>{{ feature_table|safe }}
<h2>Data and preprocessing gates</h2>{{ gate_table|safe }}
<h2>Primary analysis</h2>
{% if analysis.ready %}
<p>The configured {{ "exploratory" if development else "full-cohort" }} analysis completed for {{ analysis.clinical_n }} clinical participants using {{ analysis.features|length }} circuit features and {{ analysis.healthy_n }} healthy-reference participants.</p>
{% if analysis_conclusion %}<div class="callout"><strong>Current statistical interpretation.</strong> {{ analysis_conclusion }}</div>{% endif %}
{% for figure in analysis_figures %}<div class="figure"><img src="{{ figure.uri }}"><div class="caption">{{ figure.caption }}</div></div>{% endfor %}
{{ clinical_table|safe }}
{% else %}
<div class="callout"><strong>Analysis gated.</strong> {{ analysis.reasons|join(', ') }}. No cluster assignments or inferential results were generated.</div>
{% endif %}
<h2>Method fidelity</h2><table><tr><th>Element</th><th>Status</th><th>Implementation</th></tr>
<tr><td>Distance and linkage</td><td>Exact</td><td>Pearson correlation distance; average-linkage hierarchical clustering.</td></tr>
<tr><td>Healthy norm</td><td>Exact framework</td><td>Scanner-harmonized features expressed in healthy-reference SD units.</td></tr>
<tr><td>Task features</td><td>Adapted</td><td>FOR2107 FM/AP contrasts must be finalized from event documentation.</td></tr>
<tr><td>Cognitive control</td><td>Unavailable</td><td>No Go-NoGo equivalent; ENC is reported only as exploratory memory.</td></tr>
<tr><td>Validation</td><td>Implemented, gated</td><td>Elbow, silhouette nulls, permutation, LOO, 80% resampling, split-half.</td></tr></table>
<h2>Provenance</h2><pre><code>{{ provenance }}</code></pre>
<footer>FOR2107 adaptation of Tozzi et al. (2024) · generated outputs are deidentified · source data were not modified.</footer>
</body></html>"""


def _image_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _figure(
    fig: plt.Figure,
    caption: str,
    *,
    output_dir: Path | None = None,
    filename: str | None = None,
) -> dict[str, str]:
    """Embed a vector figure and optionally retain a standalone SVG copy."""
    if output_dir is not None and filename is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, format="svg", bbox_inches="tight")
    result = {"uri": _image_uri(fig), "caption": caption}
    if filename is not None:
        result["filename"] = filename
    return result


def _mean_confidence_interval(values: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(clean):
        return np.nan, np.nan
    mean = float(np.mean(clean))
    if len(clean) < 2:
        return mean, np.nan
    sem = float(np.std(clean, ddof=1) / np.sqrt(len(clean)))
    half_width = float(t.ppf((1 + confidence) / 2, len(clean) - 1) * sem)
    return mean, half_width


def _wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    half_width = (
        z * np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    )
    return float(max(0, center - half_width)), float(min(1, center + half_width))


def _bootstrap_median_interval(
    values: pd.Series,
    *,
    seed: int,
    resamples: int = 2000,
) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(clean):
        return np.nan, np.nan, np.nan
    median = float(np.median(clean))
    if len(clean) < 2:
        return median, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty(resamples, dtype=float)
    batch_size = 250
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        samples = rng.choice(clean, size=(stop - start, len(clean)), replace=True)
        bootstrapped[start:stop] = np.median(samples, axis=1)
    low, high = np.quantile(bootstrapped, [0.025, 0.975])
    return median, float(low), float(high)


def _clean_feature_label(name: str) -> str:
    code = name.split("_", maxsplit=1)[0]
    if code in EDGE_LABELS:
        return EDGE_LABELS[code]
    return name.removesuffix("_connectivity").replace("_", " ").title()


def _correct_legacy_empirical_p_values(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Correct old zero-capable Monte Carlo p-values for report rendering."""
    result = evaluation.copy()
    specifications = (
        ("p_permuted", "permutations", "permutation_exceedances"),
        ("p_multinormal", "simulations", "multinormal_exceedances"),
    )
    for p_column, trial_column, count_column in specifications:
        if p_column not in result or trial_column not in result:
            continue
        if count_column in result:
            counts = result[count_column].astype(int)
        else:
            counts = np.rint(result[p_column] * result[trial_column]).astype(int)
            result[count_column] = counts
        result[p_column] = [
            empirical_p_value(int(count), int(trials))
            for count, trials in zip(counts, result[trial_column], strict=True)
        ]
    return result


def _analysis_conclusion(config: Config, analysis: dict[str, Any]) -> str | None:
    if not analysis.get("ready"):
        return None
    root = config.require_path("outputs.root") / "analysis"
    selected = int(analysis["selected_clusters"])
    parts = [
        f"The pipeline produced {selected} groups because k={selected} was prespecified; "
        "this alone is not evidence that the population contains that many natural groups."
    ]
    selection_path = root / "solution_selection_status.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        supported = selection.get("supported_candidates", [])
        if selection.get("confirmatory_passes_all"):
            parts.append(
                f"The confirmatory k={selected} solution passed all four prespecified null-model and stability gates."
            )
        else:
            parts.append(
                f"The confirmatory k={selected} solution did not pass all four prespecified null-model and stability gates."
            )
        parts.append(
            "Candidate solutions passing every gate: "
            + (", ".join(f"k={value}" for value in supported) if supported else "none")
            + "."
        )
    evaluation_path = root / "solution_evaluation.tsv"
    if evaluation_path.exists():
        evaluation = _correct_legacy_empirical_p_values(pd.read_csv(evaluation_path, sep="\t"))
        row = evaluation.loc[evaluation["n_clusters"] == selected]
        if not row.empty:
            item = row.iloc[0]
            supported = item["p_permuted"] < 0.05 and item["p_multinormal"] < 0.05
            verdict = (
                "exceeded both null models"
                if supported
                else "did not exceed both null models and therefore is not stronger than chance by the prespecified criterion"
            )
            parts.append(
                f"Its silhouette was {item['silhouette']:.3f}; the solution {verdict} "
                f"(feature-permutation p={item['p_permuted']:.4f}, "
                f"multivariate-normal p={item['p_multinormal']:.4f})."
            )
    loo_path = root / "loo_stability.tsv"
    subsample_path = root / "subsample_stability.tsv"
    if loo_path.exists() and subsample_path.exists():
        loo = pd.read_csv(loo_path, sep="\t")
        subsample = pd.read_csv(subsample_path, sep="\t")
        loo_row = loo.loc[loo["n_clusters"] == selected]
        sub_row = subsample.loc[subsample["n_clusters"] == selected]
        if not loo_row.empty and not sub_row.empty:
            parts.append(
                f"Assignments were insensitive to removing one participant "
                f"(ARI {loo_row.iloc[0]['loo_ari_mean']:.3f}±{loo_row.iloc[0]['loo_ari_sd']:.3f}) "
                f"but only moderately stable in 80% subsamples "
                f"(ARI {sub_row.iloc[0]['subsample_ari_mean']:.3f}±{sub_row.iloc[0]['subsample_ari_sd']:.3f})."
            )
    clinical_path = root / "clinical_results.tsv"
    if clinical_path.exists():
        clinical = pd.read_csv(clinical_path, sep="\t")
        significant = int((clinical["p_fdr"] < 0.05).sum()) if "p_fdr" in clinical else 0
        parts.append(
            f"After FDR correction, {significant} of {len(clinical)} cluster–symptom comparisons were significant."
        )
    nuisance_path = root / "nuisance_associations.tsv"
    if nuisance_path.exists():
        nuisance = pd.read_csv(nuisance_path, sep="\t")
        significant = int((nuisance["p_fdr"] < 0.05).sum()) if "p_fdr" in nuisance else 0
        parts.append(
            f"Nuisance checks found {significant} of {len(nuisance)} site/scanner/demographic associations after FDR correction."
        )
    parts.append(
        "The present groups are best interpreted as descriptive resting-state profile patterns, not validated biotypes."
    )
    return " ".join(parts)


def _finish_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    ax.set_axisbelow(True)


def _inventory_figure(scans: pd.DataFrame) -> dict[str, str] | None:
    if scans.empty:
        return None
    func = scans.loc[scans["datatype"] == "func"]
    counts = func.groupby("task", dropna=False)["subject"].nunique().sort_values(ascending=False)
    if counts.empty:
        return None
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    palette = ["#176b87", "#19a7a0", "#5cb8b2", "#d99022"][: len(counts)]
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        hue=counts.index.astype(str),
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set(
        xlabel="Paradigm",
        ylabel="Participants with baseline scan",
        title="Baseline functional MRI availability",
    )
    ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(counts.values):
        ax.text(
            index, value + max(counts.values) * 0.02, str(value), ha="center", fontweight="bold"
        )
    return {
        "uri": _image_uri(fig),
        "caption": "Unique participants with each functional paradigm in the staged baseline inventory.",
    }


def _feature_figure(features: pd.DataFrame) -> dict[str, str] | None:
    if features.empty:
        return None
    counts = features.groupby(["source_task", "usable"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    colors = ["#d99022", "#19a7a0"] if len(counts.columns) == 2 else ["#176b87"]
    counts.plot(kind="barh", stacked=True, color=colors, ax=ax)
    ax.set(xlabel="Configured features", ylabel="Source task", title="Feature mapping readiness")
    ax.spines[["top", "right"]].set_visible(False)
    return {
        "uri": _image_uri(fig),
        "caption": "Task mappings remain gated until contrasts and event files are available; ROI files are acquired separately with checksums.",
    }


def _smoke_feature_figure(
    extracted: pd.DataFrame,
    names: list[str],
    output_dir: Path | None = None,
) -> dict[str, str] | None:
    if extracted.empty or not names:
        return None
    summaries = [
        (name, *_mean_confidence_interval(extracted[name])) for name in names if name in extracted
    ]
    summaries = [item for item in summaries if np.isfinite(item[1])]
    if not summaries:
        return None
    labels = [_clean_feature_label(name) for name, _, _ in summaries]
    means = np.asarray([mean for _, mean, _ in summaries])
    errors = np.asarray([error for _, _, error in summaries])
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    bars = ax.barh(
        labels,
        means,
        xerr=errors,
        color=SERIES_COLORS[: len(means)],
        error_kw={"ecolor": COLORS["ink"], "capsize": 4, "elinewidth": 1.2},
    )
    ax.axvline(0, color=COLORS["slate"], linewidth=0.8)
    ax.set(xlabel="Mean pairwise connectivity (Fisher z)", title="Resting-state circuit summary")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.bar_label(bars, fmt="%.3f", padding=7, fontweight="bold")
    return _figure(
        fig,
        (
            f"Mean circuit summaries across {len(extracted)} imported participants; "
            "error bars are 95% confidence intervals for the participant mean."
        ),
        output_dir=output_dir,
        filename="resting_state_extraction_summary.svg",
    )


def _analysis_figures(
    config: Config,
    analysis: dict[str, Any],
    output_dir: Path | None = None,
) -> list[dict[str, str]]:
    if not analysis.get("ready"):
        return []
    root = config.require_path("outputs.root") / "analysis"
    figures: list[dict[str, str]] = []
    assignments_path = root / "cluster_assignments.tsv"
    assignments = (
        pd.read_csv(assignments_path, sep="\t") if assignments_path.exists() else pd.DataFrame()
    )
    feature_names = [column for column in assignments if column not in {"subject_id", "cluster"}]
    evaluation_path = root / "solution_evaluation.tsv"
    if evaluation_path.exists():
        evaluation = _correct_legacy_empirical_p_values(pd.read_csv(evaluation_path, sep="\t"))
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
        axes[0].plot(
            evaluation["n_clusters"],
            evaluation["within_cluster_distance"],
            marker="o",
            color=COLORS["blue"],
        )
        axes[0].set(xlabel="Clusters", ylabel="Within-cluster distance", title="Elbow analysis")
        silhouette_errors = np.zeros(len(evaluation), dtype=float)
        if feature_names:
            matrix = assignments[feature_names].to_numpy(dtype=float)
            for index, candidate in enumerate(evaluation["n_clusters"].astype(int)):
                fit = fit_clusters(matrix, candidate)
                participant_values = silhouette_samples(
                    fit.distance_matrix,
                    fit.labels,
                    metric="precomputed",
                )
                _, silhouette_errors[index] = _mean_confidence_interval(
                    pd.Series(participant_values)
                )
        axes[1].errorbar(
            evaluation["n_clusters"],
            evaluation["silhouette"],
            yerr=silhouette_errors,
            fmt="o-",
            color=COLORS["teal"],
            capsize=4,
        )
        axes[1].set(xlabel="Clusters", ylabel="Mean silhouette", title="Cluster separation")
        for axis in axes:
            _finish_axis(axis)
        figures.append(
            _figure(
                fig,
                "Candidate solutions evaluated with the same correlation-distance hierarchy used for final clustering. Silhouette error bars are 95% intervals across participant-level silhouette contributions, conditional on each fitted solution; the aggregate elbow statistic has no equivalent conventional error bar.",
                output_dir=output_dir,
                filename="cluster_solution_overview.svg",
            )
        )

        probability_columns = [
            (
                "p_permuted",
                "Feature permutation",
                "permutations",
                "permutation_exceedances",
                COLORS["blue"],
            ),
            (
                "p_multinormal",
                "Multivariate-normal null",
                "simulations",
                "multinormal_exceedances",
                COLORS["amber"],
            ),
        ]
        if all(column in evaluation for column, _, _, _, _ in probability_columns):
            fig, ax = plt.subplots(figsize=(9.2, 3.8))
            x = evaluation["n_clusters"].to_numpy(dtype=float)
            for offset, (column, label, trials_column, count_column, color) in zip(
                (-0.06, 0.06), probability_columns, strict=True
            ):
                estimates = evaluation[column].to_numpy(dtype=float)
                trials = evaluation[trials_column].to_numpy(dtype=int)
                intervals = [
                    _wilson_interval(int(count), int(n))
                    for count, n in zip(evaluation[count_column], trials, strict=True)
                ]
                lower = np.asarray(
                    [value - low for value, (low, _) in zip(estimates, intervals, strict=True)]
                )
                upper = np.asarray(
                    [high - value for value, (_, high) in zip(estimates, intervals, strict=True)]
                )
                ax.errorbar(
                    x + offset,
                    estimates,
                    yerr=np.vstack([lower, upper]),
                    fmt="o-",
                    color=color,
                    capsize=4,
                    linewidth=1.6,
                    markersize=5,
                    label=label,
                )
            ax.axhline(
                0.05,
                color=COLORS["slate"],
                linestyle="--",
                linewidth=1,
                label="p = 0.05",
            )
            ax.set(
                xticks=x,
                xlabel="Clusters",
                ylabel="Null-test probability",
                title="Is the observed separation stronger than chance?",
                ylim=(-0.02, 1.02),
            )
            ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
            _finish_axis(ax)
            figures.append(
                _figure(
                    fig,
                    "Observed silhouette compared with two null models. Error bars are 95% Wilson intervals for Monte Carlo uncertainty; they do not represent uncertainty from recruiting a new cohort.",
                    output_dir=output_dir,
                    filename="cluster_null_probabilities.svg",
                )
            )

    selection_path = root / "solution_selection.tsv"
    if selection_path.exists():
        selection = pd.read_csv(selection_path, sep="\t")
        columns = [
            "passes_multinormal",
            "passes_permutation",
            "passes_loo",
            "passes_subsample",
        ]
        if not selection.empty and all(column in selection for column in columns):
            matrix = selection.set_index("n_clusters")[columns].astype(int)
            fig, ax = plt.subplots(figsize=(8.2, 4.7))
            sns.heatmap(
                matrix,
                cmap=sns.color_palette(["#f4c7c3", "#b9e3d7"], as_cmap=True),
                vmin=0,
                vmax=1,
                cbar=False,
                linewidths=1,
                linecolor="white",
                annot=matrix.replace({0: "×", 1: "✓"}),
                fmt="",
                ax=ax,
            )
            ax.set_xticklabels(
                ["Normal null", "Permutation null", "Leave-one-out", "80% subsamples"],
                rotation=20,
                ha="right",
            )
            ax.set(
                xlabel="Prespecified criterion",
                ylabel="Candidate k",
                title="Cluster-solution decision matrix",
            )
            confirmatory = int(analysis["selected_clusters"])
            if confirmatory in matrix.index:
                row = list(matrix.index).index(confirmatory)
                ax.add_patch(
                    plt.Rectangle(
                        (0, row),
                        len(columns),
                        1,
                        fill=False,
                        edgecolor=COLORS["ink"],
                        linewidth=2.2,
                    )
                )
            figures.append(
                _figure(
                    fig,
                    f"Green cells pass the prespecified criterion; the outlined row is the confirmatory k={confirmatory} solution. Passing these gates is necessary but not sufficient for a validated biotype solution.",
                    output_dir=output_dir,
                    filename="cluster_selection_matrix.svg",
                )
            )

    loo_path = root / "loo_stability.tsv"
    subsample_path = root / "subsample_stability.tsv"
    if loo_path.exists() and subsample_path.exists():
        loo = pd.read_csv(loo_path, sep="\t")
        subsample = pd.read_csv(subsample_path, sep="\t")
        stability = loo.merge(subsample, on="n_clusters", how="inner")
        if not stability.empty:
            fig, ax = plt.subplots(figsize=(9.2, 3.8))
            x = stability["n_clusters"].to_numpy(dtype=float)
            ax.errorbar(
                x - 0.06,
                stability["loo_ari_mean"],
                yerr=stability["loo_ari_sd"],
                fmt="o-",
                color=COLORS["blue"],
                capsize=4,
                linewidth=1.6,
                label="Leave one participant out",
            )
            ax.errorbar(
                x + 0.06,
                stability["subsample_ari_mean"],
                yerr=stability["subsample_ari_sd"],
                fmt="o-",
                color=COLORS["teal"],
                capsize=4,
                linewidth=1.6,
                label=f"{int(round(stability['fraction'].iloc[0] * 100))}% participant subsamples",
            )
            ax.set(
                xticks=x,
                xlabel="Clusters",
                ylabel="Adjusted Rand index",
                title="Assignment stability under participant perturbation",
                ylim=(-0.02, 1.08),
            )
            ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
            _finish_axis(ax)
            figures.append(
                _figure(
                    fig,
                    "Mean agreement with full-data assignments; error bars show ±1 SD across omitted participants or repeated subsamples.",
                    output_dir=output_dir,
                    filename="cluster_stability.svg",
                )
            )

    split_path = root / "split_half_summary.tsv"
    if split_path.exists():
        split = pd.read_csv(split_path, sep="\t")
        if not split.empty:
            fig, ax = plt.subplots(figsize=(8.6, 3.8))
            x = split["cluster"].to_numpy(dtype=float)
            mean = split["mean"].to_numpy(dtype=float)
            ax.errorbar(
                x,
                mean,
                yerr=np.vstack([mean - split["ci_2_5"], split["ci_97_5"] - mean]),
                fmt="o",
                color=COLORS["blue"],
                ecolor=COLORS["ink"],
                capsize=4,
            )
            ax.axhline(0, color=COLORS["slate"], linewidth=0.8)
            ax.set(
                xticks=x,
                xlabel="Cluster profile",
                ylabel="Profile correlation",
                title="Repeated split-half profile replication",
                ylim=(-1.05, 1.05),
            )
            _finish_axis(ax)
            figures.append(
                _figure(
                    fig,
                    "Points show mean matched-profile correlations across repeated random halves; error bars span the empirical 2.5th–97.5th percentiles.",
                    output_dir=output_dir,
                    filename="split_half_replication.svg",
                )
            )

    if (
        not assignments.empty
        and evaluation_path.exists()
        and loo_path.exists()
        and subsample_path.exists()
    ):
        evaluation = _correct_legacy_empirical_p_values(pd.read_csv(evaluation_path, sep="\t"))
        loo = pd.read_csv(loo_path, sep="\t")
        subsample = pd.read_csv(subsample_path, sep="\t")
        replicated = supplementary_clustering_figures(
            assignments,
            evaluation,
            loo,
            subsample,
            confirmatory=int(analysis["selected_clusters"]),
            seed=int(config.get("project.random_seed", 2107)),
        )
        for stem, (fig, caption, artifacts) in replicated.items():
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_dir / f"{stem}.pdf", format="pdf", bbox_inches="tight")
                fig.savefig(output_dir / f"{stem}.png", format="png", dpi=300, bbox_inches="tight")
                for artifact_name, artifact in artifacts.items():
                    if isinstance(artifact, dict):
                        for panel_name, table in artifact.items():
                            safe_panel = panel_name.lower().replace(" ", "_")
                            table.to_csv(
                                output_dir / f"{stem}_{artifact_name}_{safe_panel}.tsv",
                                sep="\t",
                                index=False,
                            )
            figures.append(
                _figure(
                    fig,
                    caption,
                    output_dir=output_dir,
                    filename=f"{stem}.svg",
                )
            )

    site_path = root / "leave_site_out.tsv"
    if site_path.exists():
        site = pd.read_csv(site_path, sep="\t")
        if not site.empty:
            pairs = site["training_site"].astype(str) + " → " + site["test_site"].astype(str)
            plot = site.assign(site_pair=pairs)
            fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
            sns.boxplot(
                data=plot, x="site_pair", y="profile_correlation", color="#b9e3d7", ax=axes[0]
            )
            ari = plot.groupby("site_pair", as_index=False)[
                "assignment_ari_vs_independent_test_clustering"
            ].first()
            sns.barplot(
                data=ari,
                x="site_pair",
                y="assignment_ari_vs_independent_test_clustering",
                color=COLORS["blue"],
                ax=axes[1],
            )
            axes[0].set(
                title="Transferred profile similarity",
                xlabel="Training → test site",
                ylabel="Profile correlation",
                ylim=(-1.05, 1.05),
            )
            axes[1].set(
                title="Transferred assignment agreement",
                xlabel="Training → test site",
                ylabel="Adjusted Rand index",
                ylim=(-0.05, 1.05),
            )
            for ax in axes:
                ax.tick_params(axis="x", rotation=25)
                _finish_axis(ax)
            figures.append(
                _figure(
                    fig,
                    "Leave-site-out transfer after independent within-site healthy referencing. Profile correlations summarize the six transferred groups; ARI compares transferred labels with clustering performed independently at the test site.",
                    output_dir=output_dir,
                    filename="leave_site_out_replication.svg",
                )
            )
    linkage_path = root / "linkage_matrix.npy"
    if linkage_path.exists():
        fig, ax = plt.subplots(figsize=(9.2, 4.2))
        dendrogram(
            np.load(linkage_path),
            color_threshold=None,
            no_labels=True,
            above_threshold_color=COLORS["slate"],
            ax=ax,
        )
        ax.set(
            xlabel="Participants", ylabel="Correlation distance", title="Average-linkage hierarchy"
        )
        _finish_axis(ax)
        figures.append(
            _figure(
                fig,
                "Hierarchical clustering of clinical circuit-score profiles; leaf labels are suppressed for privacy.",
                output_dir=output_dir,
                filename="cluster_hierarchy.svg",
            )
        )

    if not assignments.empty:
        grouped = assignments.groupby("cluster")
        profiles = grouped[feature_names].mean()
        intervals = grouped[feature_names].agg(lambda values: _mean_confidence_interval(values)[1])
        labels = [_clean_feature_label(name) for name in feature_names]
        if len(feature_names) == 14:
            deviations = cluster_deviation_tests(assignments, feature_names)
            fig = tozzi_style_rest_figure(
                assignments,
                deviation_tests=deviations,
            )
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                deviations.to_csv(
                    output_dir / "tozzi_style_resting_profiles_statistics.tsv",
                    sep="\t",
                    index=False,
                )
                fig.savefig(
                    output_dir / "tozzi_style_resting_profiles.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )
                fig.savefig(
                    output_dir / "tozzi_style_resting_profiles.png",
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                )
            figures.append(
                _figure(
                    fig,
                    (
                        "Rest-only counterpart of Tozzi et al. Fig. 3. Gray points are participants; "
                        "bars are cluster means with unadjusted 95% confidence intervals. Circuit colors indicate two-sided "
                        "deviations from healthy-reference zero surviving BH-FDR correction across all cluster-edge "
                        "tests; non-significant deviations remain gray. Solid and dashed schematic connections "
                        "mark means above +0.5 or below -0.5 healthy-reference SD. These significance tests are "
                        "post-clustering descriptions, not independent confirmation. "
                        "Rows are neutrally labeled because these profiles do not establish the published biotypes."
                    ),
                    output_dir=output_dir,
                    filename="tozzi_style_resting_profiles.svg",
                )
            )
        fig, ax = plt.subplots(figsize=(10.8, 5.0))
        if len(feature_names) > 6:
            x = np.arange(len(feature_names))
            for index, cluster in enumerate(profiles.index):
                ax.errorbar(
                    x,
                    profiles.loc[cluster, feature_names],
                    yerr=intervals.loc[cluster, feature_names],
                    fmt="o-",
                    linewidth=1.2,
                    markersize=4,
                    capsize=2,
                    label=f"Cluster {cluster}",
                    color=SERIES_COLORS[index % len(SERIES_COLORS)],
                )
            ax.set_xticks(x, labels, rotation=45, ha="right")
            xlabel = "Imaging feature"
        else:
            x = np.arange(len(profiles))
            width = 0.8 / len(feature_names)
            for index, (feature, label) in enumerate(zip(feature_names, labels, strict=True)):
                offset = (index - (len(feature_names) - 1) / 2) * width
                ax.bar(
                    x + offset,
                    profiles[feature],
                    yerr=intervals[feature],
                    width=width,
                    label=label,
                    color=SERIES_COLORS[index % len(SERIES_COLORS)],
                    error_kw={"ecolor": COLORS["ink"], "capsize": 3, "elinewidth": 1.1},
                )
            ax.set_xticks(x, [f"Cluster {value}" for value in profiles.index])
            xlabel = "Exploratory cluster"
        ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
        ax.axhline(0.5, color="#94a3b8", linewidth=0.8, linestyle="--")
        ax.axhline(-0.5, color="#94a3b8", linewidth=0.8, linestyle="--")
        ax.set(
            xlabel=xlabel,
            ylabel="Mean healthy-reference SD",
            title="All imaging-feature profiles by exploratory cluster",
        )
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
        _finish_axis(ax)
        figures.append(
            _figure(
                fig,
                (
                    "All-feature analogue of the column plots in Tozzi et al. Fig. 3. "
                    "Lines show cluster means and error bars are 95% confidence intervals for those means. "
                    "Published resting-state edges and adapted task-derived features are shown together; "
                    "dashed lines mark ±0.5 healthy-reference SD. This exploratory result does not by itself "
                    "establish the published multimodal biotypes."
                ),
                output_dir=output_dir,
                filename="cluster_circuit_profiles.svg",
            )
        )

        if len(feature_names) <= 6:
            long_assignments = assignments.melt(
                id_vars=["subject_id", "cluster"],
                value_vars=feature_names,
                var_name="circuit",
                value_name="score",
            )
            long_assignments["circuit"] = long_assignments["circuit"].map(_clean_feature_label)
            fig, ax = plt.subplots(figsize=(9.2, 4.5))
            sns.boxplot(
                data=long_assignments,
                x="cluster",
                y="score",
                hue="circuit",
                palette=SERIES_COLORS[: len(feature_names)],
                showfliers=False,
                width=0.75,
                ax=ax,
            )
            ax.axhline(0, color=COLORS["slate"], linewidth=0.8)
            ax.set(
                xlabel="Exploratory cluster",
                ylabel="Healthy-reference SD",
                title="Participant-level variation within each circuit profile",
            )
            ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
            _finish_axis(ax)
            figures.append(
                _figure(
                    fig,
                    "Boxes show medians and interquartile ranges; whiskers extend to 1.5×IQR. This view reveals within-cluster overlap that mean profiles alone conceal.",
                    output_dir=output_dir,
                    filename="cluster_circuit_distributions.svg",
                )
            )

        fig, ax = plt.subplots(figsize=(9.2, max(3.0, 0.55 * len(profiles))))
        sns.heatmap(
            profiles.rename(columns=_clean_feature_label),
            cmap=sns.diverging_palette(220, 25, as_cmap=True),
            center=0,
            annot=len(feature_names) <= 8,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Healthy-reference SD"},
            ax=ax,
        )
        ax.set(
            xlabel="Circuit feature", ylabel="Cluster", title="Mean circuit dysfunction profiles"
        )
        figures.append(
            _figure(
                fig,
                "Cluster-average regional circuit scores in healthy-reference standard-deviation units.",
                output_dir=output_dir,
                filename="cluster_profile_heatmap.svg",
            )
        )

        participants_path = config.require_path("outputs.root") / "cohort" / "participants.tsv"
        if participants_path.exists():
            participants = pd.read_csv(participants_path, sep="\t")
            outcomes = [
                name
                for name in ("bdi_total", "hama_total", "hamd17_total", "hamd21_total")
                if name in participants and participants[name].notna().any()
            ]
            clinical = assignments[["subject_id", "cluster"]].merge(
                participants[["subject_id", *outcomes]], on="subject_id", how="left"
            )
            if outcomes:
                titles = {
                    "bdi_total": "BDI depression",
                    "hama_total": "HAM-A anxiety",
                    "hamd17_total": "HAM-D 17-item",
                    "hamd21_total": "HAM-D 21-item",
                }
                fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex=True)
                for measure_index, (ax, measure) in enumerate(
                    zip(axes.flat, outcomes, strict=False)
                ):
                    rows = []
                    for cluster in sorted(clinical["cluster"].dropna().unique()):
                        median, low, high = _bootstrap_median_interval(
                            clinical.loc[clinical["cluster"] == cluster, measure],
                            seed=(
                                int(config.get("project.random_seed", 2107))
                                + measure_index * 100
                                + int(cluster)
                            ),
                        )
                        rows.append((cluster, median, low, high))
                    cluster_values = np.asarray([row[0] for row in rows])
                    medians = np.asarray([row[1] for row in rows])
                    lower = medians - np.asarray([row[2] for row in rows])
                    upper = np.asarray([row[3] for row in rows]) - medians
                    ax.errorbar(
                        cluster_values,
                        medians,
                        yerr=np.vstack([lower, upper]),
                        fmt="o",
                        color=SERIES_COLORS[measure_index % len(SERIES_COLORS)],
                        ecolor=COLORS["ink"],
                        capsize=4,
                        markersize=6,
                    )
                    ax.set(title=titles.get(measure, measure), ylabel="Median score")
                    _finish_axis(ax)
                for ax in axes[-1]:
                    ax.set_xlabel("Exploratory cluster")
                    ax.set_xticks(sorted(clinical["cluster"].dropna().unique()))
                fig.suptitle("Clinical symptom distributions by cluster", y=1.01, fontweight="bold")
                fig.tight_layout()
                figures.append(
                    _figure(
                        fig,
                        "Points show cluster medians and error bars show participant-bootstrap 95% confidence intervals. These are descriptive summaries; multiplicity-adjusted tests remain in the results table.",
                        output_dir=output_dir,
                        filename="cluster_clinical_outcomes.svg",
                    )
                )
    return figures


def render_report(
    config: Config,
    summary: dict[str, Any],
    scans: pd.DataFrame,
    features: pd.DataFrame,
    preflight: dict[str, Any],
    provenance: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, str | None]:
    output_root = config.require_path("outputs.reports")
    output_root.mkdir(parents=True, exist_ok=True)
    figure_dir = output_root / "figures"
    figures = [item for item in (_inventory_figure(scans), _feature_figure(features)) if item]
    extracted_path = config.require_path("outputs.root") / "features" / "regional_features.tsv"
    extracted = pd.read_csv(extracted_path, sep="\t") if extracted_path.exists() else pd.DataFrame()
    usable_names = (
        features.loc[features["usable"].astype(bool), "name"].tolist() if not features.empty else []
    )
    available_names = [
        name for name in usable_names if name in extracted and extracted[name].notna().any()
    ]
    smoke_rows = []
    for name in available_names:
        values = pd.to_numeric(extracted[name], errors="coerce")
        mean, half_width = _mean_confidence_interval(values)
        smoke_rows.append(
            {
                "Circuit": _clean_feature_label(name),
                "N": int(values.notna().sum()),
                "Mean Fisher z": f"{mean:.3f}",
                "SD": f"{values.std(ddof=1):.3f}",
                "95% CI": f"[{mean - half_width:.3f}, {mean + half_width:.3f}]",
                "Missing": int(values.isna().sum()),
            }
        )
    smoke_display = pd.DataFrame(smoke_rows)
    feature_columns = ["name", "source_task", "modality", "equivalence", "usable", "validation"]
    feature_table = (
        features[feature_columns].to_html(index=False, border=0)
        if not features.empty
        else "<p>No feature definitions found.</p>"
    )
    task_inputs_present = bool(
        not scans.empty and "task" in scans and scans["task"].fillna("").ne("rest").any()
    )
    gate_rows = [
        {"Gate": "Python environment", "Ready": all(preflight["python_modules"].values())},
        {"Gate": "Preprocessing backend", "Ready": preflight["backend_ready"]},
        {
            "Gate": "Motion QC",
            "Ready": bool(len(extracted))
            and "motion_qc_available" in extracted
            and bool(extracted["motion_qc_available"].fillna(False).all()),
        },
        {
            "Gate": "Task inputs and events",
            "Ready": task_inputs_present and summary["missing_task_events"] == 0,
        },
        {
            "Gate": "Feature definitions",
            "Ready": bool(len(features)) and bool(features["usable"].all()),
        },
        {"Gate": "Scientific sample size", "Ready": config.get("project.mode") == "full"},
    ]
    status = "Inventory and validation completed. " + (
        f"Connectivity extraction succeeded for {len(extracted)} participant(s); full feature extraction is gated by the items below."
        if any(not row["Ready"] for row in gate_rows)
        else "Inputs are ready for preprocessing and feature extraction."
    )
    environment = Environment(autoescape=select_autoescape(["html", "xml"]))
    clinical_path = config.require_path("outputs.root") / "analysis" / "clinical_results.tsv"
    clinical_table = (
        pd.read_csv(clinical_path, sep="\t").to_html(index=False, border=0)
        if analysis.get("ready") and clinical_path.exists()
        else ""
    )
    html = environment.from_string(REPORT_TEMPLATE).render(
        title=config.get("report.title"),
        subtitle=(
            "Resting-state methodological adaptation"
            if not task_inputs_present
            else "Baseline multimodal methodological reproduction"
        ),
        mode=config.get("project.mode"),
        development=config.get("project.mode") != "full",
        created=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        summary=summary,
        status=status,
        limitations=summary.get("limitations", []),
        figures=figures,
        smoke_figure=_smoke_feature_figure(extracted, available_names, figure_dir),
        smoke_table=smoke_display.to_html(index=False, border=0) if not smoke_display.empty else "",
        feature_table=feature_table,
        gate_table=pd.DataFrame(gate_rows).to_html(index=False, border=0),
        analysis=analysis,
        analysis_conclusion=_analysis_conclusion(config, analysis),
        analysis_figures=_analysis_figures(config, analysis, figure_dir),
        clinical_table=clinical_table,
        provenance=json.dumps(provenance, indent=2),
    )
    html_path = output_root / "for2107_tozzi_report.html"
    html_path.write_text(html, encoding="utf-8")
    pdf_path: Path | None = output_root / "for2107_tozzi_report.pdf"
    try:
        from weasyprint import HTML

        HTML(string=html, base_url=str(output_root)).write_pdf(pdf_path)
    except (ImportError, OSError):
        pdf_path = None
    return {"html": str(html_path), "pdf": str(pdf_path) if pdf_path else None}
