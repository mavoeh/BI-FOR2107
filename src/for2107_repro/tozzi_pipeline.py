from __future__ import annotations

import json
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .analysis import run_primary_analysis
from .clinical import read_clinical, select_cohorts
from .clustering_figures import supplementary_clustering_figures
from .config import Config
from .progress import ProgressCounter, report, stage
from .provenance import write_provenance
from .tozzi_method import (
    TOZZI_ACTIVATION_NETWORKS,
    TOZZI_PPI_EDGES,
    TOZZI_TASK_FREE_CIRCUITS,
    discover_task_pairs,
    extract_tozzi_subject,
    extraction_rois,
    roi_network,
    task_free_definitions,
)


def _feature_selection_table(config: Config, features: pd.DataFrame) -> pd.DataFrame:
    """Catalog every extracted column: runs, contrast, ROIs and Tozzi definition.

    ``equivalence`` is ``published_edge`` / ``within_network_extension`` for
    task-free edges and, for task features, ``closest_task_analogue`` (Tozzi's
    feature in the contrast configured to stand in for its circuit),
    ``tozzi_rois_other_contrast`` (Tozzi's ROIs or PPI pair, another contrast)
    or ``network_extension``. ``tozzi_circuit`` names the matched circuit.
    """
    present = set(features.columns)
    matches = config.get("tozzi.construct_matches", {})
    rows: list[dict[str, Any]] = []

    def add(name: str, **row: Any) -> None:
        if name in present:
            rows.append({"feature": name, **row})

    for definition in task_free_definitions(config):
        network = roi_network(definition.rois[0])
        published = definition.equivalence == "published_edge"
        for suffix, runs in (("", "AP+FM"), ("__AP", "AP"), ("__FM", "FM")):
            add(
                f"{definition.name}{suffix}",
                network=network,
                source_task=runs,
                contrast="task-free",
                modality="connectivity",
                rois=";".join(definition.rois),
                physiological_roi="",
                equivalence=definition.equivalence,
                tozzi_circuit=f"{TOZZI_TASK_FREE_CIRCUITS[network]}_C" if published else "",
            )

    def task_labels(network: str, matched: dict[str, str], defined: bool, modality: str) -> dict:
        circuit = matched.get(network) if defined else None
        if circuit:
            return {
                "equivalence": "closest_task_analogue",
                "tozzi_circuit": f"{circuit}_{modality}",
            }
        equivalence = "tozzi_rois_other_contrast" if defined else "network_extension"
        return {"equivalence": equivalence, "tozzi_circuit": ""}

    roi_names = extraction_rois(config)
    for task_name, contrasts in config.get("tozzi.task_contrasts", {}).items():
        for label, spec in contrasts.items():
            contrast = f"{spec['positive']} > {spec['negative']}"
            matched = matches.get(f"{task_name}_{label}", {})
            for roi in spec.get("rois", roi_names):
                network = roi_network(roi)
                add(
                    f"{task_name}_{label}_{roi}_activation_adapted",
                    network=network,
                    source_task=task_name,
                    contrast=contrast,
                    modality="activation",
                    rois=roi,
                    physiological_roi="",
                    **task_labels(network, matched, network in TOZZI_ACTIVATION_NETWORKS, "A"),
                )
            for seed, target in spec.get("ppi_edges", []):
                network = roi_network(seed)
                labels = task_labels(
                    network, matched, frozenset((seed, target)) in TOZZI_PPI_EDGES, "C"
                )
                name = f"{task_name}_{label}_{seed}__{target}_ppi_adapted"
                for suffix, physiological in (
                    ("", ""),
                    (f"__phys_{seed}", seed),
                    (f"__phys_{target}", target),
                ):
                    add(
                        f"{name}{suffix}",
                        network=network,
                        source_task=task_name,
                        contrast=contrast,
                        modality="PPI",
                        rois=f"{seed};{target}",
                        physiological_roi=physiological,
                        **labels,
                    )
    catalog = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    # Identical masks (the sad- and threat-circuit amygdalae) give identical
    # columns; point each repeat at the copy that is a Tozzi feature, if any,
    # else at the first by name.
    first: dict[bytes, str] = {}
    duplicate_of = {}
    for name in catalog.sort_values(["tozzi_circuit", "feature"], key=_tozzi_last).feature:
        seen = first.setdefault(features[name].to_numpy(float).tobytes(), name)
        duplicate_of[name] = "" if seen == name else seen
    catalog["duplicate_of"] = catalog.feature.map(duplicate_of)
    return catalog


def _tozzi_last(column: pd.Series) -> pd.Series:
    """Sort key: labelled Tozzi features first, then by name."""
    return column.eq("") if column.name == "tozzi_circuit" else column


def _generate_supplementary_figures(config: Config, root: Path) -> dict[str, Any]:
    """Generate Tozzi Supplementary Figures 1–6 from cached analysis tables."""
    analysis_root = root / "analysis"
    required = [
        analysis_root / "cluster_assignments.tsv",
        analysis_root / "solution_evaluation.tsv",
        analysis_root / "loo_stability.tsv",
        analysis_root / "subsample_stability.tsv",
    ]
    if any(not path.exists() for path in required):
        return {"ready": False, "reason": "analysis-tables-missing", "figures": []}
    assignments = pd.read_csv(required[0], sep="\t")
    evaluation = pd.read_csv(required[1], sep="\t")
    loo = pd.read_csv(required[2], sep="\t")
    subsample = pd.read_csv(required[3], sep="\t")
    if assignments.empty or evaluation.empty:
        return {"ready": False, "reason": "analysis-tables-empty", "figures": []}
    analysis_status_path = analysis_root / "status.json"
    selected = int(config.get("analysis.selected_clusters", 6))
    if analysis_status_path.exists():
        selected = int(
            json.loads(analysis_status_path.read_text(encoding="utf-8")).get(
                "selected_clusters", selected
            )
        )
    figure_dir = root / "figures" / "supplementary"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = supplementary_clustering_figures(
        assignments,
        evaluation,
        loo,
        subsample,
        confirmatory=selected,
        seed=int(config.get("project.random_seed", 2107)),
    )
    manifest: dict[str, Any] = {
        "ready": True,
        "selected_clusters": selected,
        "source_analysis": str(analysis_root),
        "figures": [],
    }
    for stem, (figure, caption, artifacts) in figures.items():
        for suffix, kwargs in (
            ("svg", {"format": "svg"}),
            ("pdf", {"format": "pdf"}),
            ("png", {"format": "png", "dpi": 300}),
        ):
            figure.savefig(figure_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
        for artifact_name, artifact in artifacts.items():
            if isinstance(artifact, dict):
                for panel_name, table in artifact.items():
                    safe_panel = panel_name.lower().replace(" ", "_")
                    table.to_csv(
                        figure_dir / f"{stem}_{artifact_name}_{safe_panel}.tsv",
                        sep="\t",
                        index=False,
                    )
        manifest["figures"].append(
            {
                "stem": stem,
                "caption": caption,
                "svg": str(figure_dir / f"{stem}.svg"),
                "pdf": str(figure_dir / f"{stem}.pdf"),
                "png": str(figure_dir / f"{stem}.png"),
            }
        )
        plt.close(figure)
    (figure_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _generate_paper_figures(config: Config, analysis: dict[str, Any], root: Path) -> dict[str, Any]:
    """Wire the existing full report figure suite into the Tozzi pipeline."""
    if not analysis.get("ready"):
        return {"ready": False, "reason": "analysis-not-ready", "figures": []}
    from .report import _analysis_figures

    figure_dir = root / "figures" / "paper"
    figure_dir.mkdir(parents=True, exist_ok=True)
    try:
        figures = _analysis_figures(config, analysis, figure_dir)
    except (KeyError, OSError, ValueError) as exc:
        return {"ready": False, "reason": f"figure-generation-failed: {exc}", "figures": []}
    manifest = {
        "ready": True,
        "source_analysis": str(root / "analysis"),
        "figures": [
            {
                "filename": item.get("filename"),
                "caption": item.get("caption", ""),
                "path": str(figure_dir / str(item.get("filename", ""))),
            }
            for item in figures
            if item.get("filename")
        ],
    }
    (figure_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _write_pdf_summary(path: Path, status: dict[str, Any]) -> None:
    """Create a dependency-light, typeset PDF when optional HTML-to-PDF is absent."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    analysis = status.get("analysis", {})
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f4f7fa")
    fig.text(
        0.09,
        0.93,
        "FOR2107  /  METHOD UPDATE",
        color="#167c80",
        fontsize=9,
        weight="bold",
        family="DejaVu Sans",
    )
    fig.text(
        0.09,
        0.885,
        "Tozzi-method task-residual analysis",
        color="#12304a",
        fontsize=23,
        weight="bold",
        family="DejaVu Sans",
    )
    fig.text(
        0.09,
        0.85,
        "Independent pipeline · preliminary implementation",
        color="#537086",
        fontsize=10,
        family="DejaVu Sans",
    )
    summary = [
        ("PAIRED AP/FM", status.get("task_pair_counts", {}).get("ready", 0)),
        ("SUBJECTS EXTRACTED", status.get("feature_rows", 0)),
        ("MDD + ANXIETY", status.get("clinical_n", "pending")),
        ("HEALTHY REF.", status.get("healthy_n", "pending")),
    ]
    for index, (label, value) in enumerate(summary):
        x = 0.09 + index * 0.205
        fig.text(x, 0.79, label, fontsize=8, color="#537086", weight="bold")
        fig.text(x, 0.745, str(value), fontsize=19, color="#12304a", weight="bold")
    fig.text(0.09, 0.68, "CURRENT INTERPRETATION", fontsize=10, weight="bold", color="#167c80")
    fig.text(
        0.09,
        0.645,
        "Method implementation and preliminary cohort analysis only — not a successful "
        "replication claim. No Go/NoGo proxy is included.",
        fontsize=10,
        color="#1f2937",
        wrap=True,
    )
    bullets = [
        "Task-free: run-wise task GLM residuals, band-pass 0.009–0.08 Hz, AP/FM concatenation, 14 Tozzi ROI-pair Fisher-z features.",
        "Confounds: six motion parameters plus mean CSF and white-matter signals; FEAT input already has 100-s high-pass and 5-mm smoothing.",
        "Task adaptations: AP sad/neutral and happy/neutral; FM faces/shapes; regional activation and reciprocal PPI features.",
        "Known method differences: FSL preprocessing; SciPy Butterworth rather than FSL filtering; PPI omits SPM deconvolution/reconvolution; FOR2107 task contrasts are not Tozzi-equivalent.",
    ]
    y = 0.57
    for bullet in bullets:
        wrapped = textwrap.fill(bullet, width=92, subsequent_indent="   ")
        fig.text(0.10, y, "•", fontsize=15, color="#df9f38", weight="bold", va="top")
        fig.text(0.13, y, wrapped, fontsize=9.3, color="#1f2937", va="top", linespacing=1.45)
        y -= 0.105 if len(wrapped) < 115 else 0.14
    fig.text(0.09, 0.12, "CLUSTERING / VALIDATION", fontsize=10, weight="bold", color="#167c80")
    if analysis.get("ready"):
        label = (
            f"Analysis completed for {analysis.get('clinical_n', 'selected')} clinical and "
            f"{analysis.get('healthy_n', 'selected')} reference participants. "
            f"Selected k={analysis.get('selected_clusters', 'n/a')}; interpret as preliminary pending replication checks."
        )
    else:
        label = "Clustering not run or gated by minimum cohort requirements; see analysis/status.json for the exact gate."
    fig.text(0.09, 0.085, textwrap.fill(label, width=98), fontsize=9, color="#1f2937", va="top")
    fig.text(
        0.09,
        0.035,
        "Machine-readable tables and full method notes are stored beside this report.",
        fontsize=8,
        color="#537086",
    )
    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _write_method_report(report_dir: Path, status: dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    analysis = status.get("analysis", {})
    task_rows = status.get("task_pair_counts", {})
    figure_items = status.get("supplementary_figures", {}).get("figures", [])
    figure_links = "".join(
        f'<li><a href="{item["png"]}">{item["stem"]}</a> — {item["caption"]}</li>'
        for item in figure_items
    )
    paper_items = status.get("paper_figures", {}).get("figures", [])
    paper_links = "".join(
        f'<li><a href="{item["path"]}">{item["filename"]}</a> — {item["caption"]}</li>'
        for item in paper_items
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Tozzi-method FOR2107 analysis</title>
<style>
body{{font:16px/1.55 'DejaVu Sans',Arial,sans-serif;color:#1f2937;background:#f4f7fa;margin:0}}main{{max-width:980px;margin:38px auto;background:white;padding:42px 54px;box-shadow:0 8px 35px #18304718}}
h1{{font-size:30px;color:#12304a;margin-bottom:6px}}h2{{font-size:19px;color:#167c80;margin-top:30px}}
.tag{{color:#537086;font-size:14px}}.callout{{border-left:4px solid #df9f38;background:#fff8e8;padding:15px 18px}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:9px 12px;border-bottom:1px solid #dbe4ea;text-align:left}}
th{{color:#456174;font-size:13px;text-transform:uppercase;letter-spacing:.04em}}
</style></head><body><main>
<h1>FOR2107 · Tozzi-method analysis</h1><div class="tag">Automated report · independent from the existing resting-state pipeline</div>
<div class="callout"><b>Interpretation:</b> this is a method implementation and preliminary cohort analysis, not yet a successful replication. Task-residual features are intended to follow Tozzi; AP/FM task contrasts and PPI are dataset-specific adaptations. Go/NoGo features remain unavailable.</div>
<h2>Run summary</h2><table><tr><th>Measure</th><th>Result</th></tr>
<tr><td>Participants with both AP and FM inputs and events</td><td>{task_rows.get("ready", 0)}</td></tr>
<tr><td>Feature rows extracted</td><td>{status.get("feature_rows", 0)}</td></tr>
<tr><td>Configured feature dimensions (14 task-free + adapted task features)</td><td>{status.get("feature_count", 0)}</td></tr>
<tr><td>Clinical cohort</td><td>{status.get("clinical_n", "not selected")}</td></tr>
<tr><td>Healthy reference cohort</td><td>{status.get("healthy_n", "not selected")}</td></tr>
<tr><td>Clustering/statistical validation</td><td>{"completed" if analysis.get("ready") else "gated / pending"}</td></tr>
</table>
<h2>Methods and known differences</h2><ul>
<li>Task-free connectivity is derived from residuals of affective-priming and face-matching task GLMs, filtered separately at 0.009–0.08 Hz, then concatenated before the 14 configured published edge correlations.</li>
<li>Task GLMs use an SPM canonical HRF, six motion parameters, and task regressors. Existing FEAT inputs already have 100-second high-pass filtering and 5-mm smoothing; these are not repeated.</li>
<li>Residual band-pass currently uses a second-order zero-phase Butterworth filter in SciPy, not FSL's exact temporal-filter implementation.</li>
<li>Activation and PPI are task-aligned FOR2107 adaptations. PPI uses a first principal-component seed series and omits SPM PPI deconvolution/reconvolution.</li>
<li>CSF and white-matter mean-signal regressors are included along with six motion parameters. The Tozzi Go/NoGo circuit and exact task contrasts are unavailable in FOR2107.</li>
<li>FEAT's functional-to-MNI registration is used for ROI sampling; transform/mask coverage failures exclude that run/participant and are written to the extraction audit.</li>
</ul><p>See <code>features/</code>, <code>cohort/</code>, and <code>analysis/</code> for machine-readable outputs.</p>
<h2>Supplementary-style figures</h2>
<p>Figures 1–6 are generated from the cached clustering tables. PNG links are shown below; SVG and PDF versions, plus split-half artifact tables, are in the figure directory.</p>
<ul>{figure_links or "<li>Figures are pending because clustering tables are not available.</li>"}</ul>
<h2>Paper-style result figures</h2>
<p>The existing repository result-figure suite is also generated from the same analysis outputs. These are saved as SVG files under <code>figures/paper/</code>.</p>
<ul>{paper_links or "<li>Paper-style figures are pending because the analysis is not ready.</li>"}</ul>
</main></body></html>"""
    path = report_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    try:
        from weasyprint import HTML

        HTML(string=html, base_url=str(report_dir)).write_pdf(str(report_dir / "report.pdf"))
    except ImportError:
        _write_pdf_summary(report_dir / "report.pdf", status)
    return path


def run_tozzi_pipeline(config: Config) -> dict[str, Any]:
    """Run independent task-residual Tozzi-method pipeline; preserve existing outputs."""
    root = config.require_path("outputs.root")
    root.mkdir(parents=True, exist_ok=True)
    pairs = discover_task_pairs(config)
    ready = pairs.loc[pairs.ready].copy() if not pairs.empty else pairs
    report(f"Discovered {len(pairs)} paired AP/FM participants; {len(ready)} have both event files")
    output_features = root / "features"
    output_features.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_features / "paired_inputs.tsv", sep="\t", index=False)

    items = list(ready.itertuples(index=False))
    rows_by_index: dict[int, dict[str, Any]] = {}
    audits: list[dict[str, Any] | None] = [None] * len(items)
    workers = max(1, int(config.get("tozzi.extraction_jobs", 6)))
    report(f"Extracting task-residual features with {workers} subject workers")
    progress = ProgressCounter("Tozzi-method feature extraction", len(items))

    def _extract(item: Any) -> dict[str, Any]:
        return extract_tozzi_subject(
            config,
            item.subject_id,
            Path(item.AP_feat),
            Path(item.FM_feat),
            Path(item.AP_events),
            Path(item.FM_events),
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_extract, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            index = futures[future]
            item = items[index]
            try:
                rows_by_index[index] = future.result()
                audits[index] = {"subject_id": item.subject_id, "ready": True, "reason": "ok"}
            except Exception as exc:
                audits[index] = {
                    "subject_id": item.subject_id,
                    "ready": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                report(f"Skipping {item.subject_id}: {exc}")
            progress.advance()
            finished = sum(audit is not None for audit in audits)
            if finished % 25 == 0 or finished == len(items):
                checkpoint = pd.DataFrame([rows_by_index[index] for index in sorted(rows_by_index)])
                checkpoint.to_csv(
                    output_features / "regional_features.partial.tsv", sep="\t", index=False
                )
                pd.DataFrame([audit for audit in audits if audit is not None]).to_csv(
                    output_features / "extraction_audit.partial.tsv", sep="\t", index=False
                )
                (output_features / "progress.json").write_text(
                    json.dumps(
                        {
                            "finished": finished,
                            "total": len(items),
                            "extracted": len(rows_by_index),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    features = pd.DataFrame(rows)
    feature_checkpoint = output_features / "regional_features.partial.tsv"
    audit_checkpoint = output_features / "extraction_audit.partial.tsv"
    if feature_checkpoint.exists():
        feature_checkpoint.replace(output_features / "regional_features.tsv")
    else:
        features.to_csv(output_features / "regional_features.tsv", sep="\t", index=False)
    if audit_checkpoint.exists():
        audit_checkpoint.replace(output_features / "extraction_audit.tsv")
    else:
        pd.DataFrame([audit for audit in audits if audit is not None]).to_csv(
            output_features / "extraction_audit.tsv", sep="\t", index=False
        )
    (output_features / "extraction_status.json").write_text(
        json.dumps(
            {
                "discovered_participant_count": len(pairs),
                "paired_input_count": int(pairs["paired"].sum()) if "paired" in pairs else 0,
                "ready_input_count": len(ready),
                "extracted_participants": len(features),
                "feature_columns": [c for c in features if c != "subject_id"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _feature_selection_table(config, features).to_csv(
        output_features / "feature_selection.tsv", sep="\t", index=False
    )

    analysis_status: dict[str, Any] = {
        "ready": False,
        "reasons": ["clinical-database-not-available"],
    }
    participants = pd.DataFrame()
    if not features.empty and config.get("inputs.primary_clinical_file"):
        with stage("Select diagnosis and healthy-reference cohorts"):
            clinical = read_clinical(config)
            cohort = select_cohorts(clinical, config)
            participants = cohort.all_participants.loc[
                cohort.all_participants.subject_id.isin(features.subject_id)
            ].copy()
            clinical_ids = set(cohort.clinical.subject_id)
            participants["analysis_role"] = participants.subject_id.map(
                lambda value: "clinical" if value in clinical_ids else "healthy_reference"
            )
            cohort_dir = root / "cohort"
            cohort_dir.mkdir(parents=True, exist_ok=True)
            participants.to_csv(cohort_dir / "participants.tsv", sep="\t", index=False)
            cohort.exclusions.to_csv(cohort_dir / "exclusions.tsv", sep="\t", index=False)
        if not config.get("analysis.run_clustering", True):
            analysis_status = {"ready": False, "reasons": ["clustering-disabled-in-config"]}
        elif len(participants):
            with stage("Run clustering and statistical validation"):
                analysis_status = run_primary_analysis(config, features, participants)
    with stage("Generate Tozzi supplementary figures"):
        figures = _generate_supplementary_figures(config, root)
    with stage("Generate paper-style result figures"):
        paper_figures = _generate_paper_figures(config, analysis_status, root)
    clinical_n = int((participants.get("analysis_role", pd.Series(dtype=str)) == "clinical").sum())
    healthy_n = int(
        (participants.get("analysis_role", pd.Series(dtype=str)) == "healthy_reference").sum()
    )
    status: dict[str, Any] = {
        "mode": "independent-tozzi-task-residual",
        "task_pair_counts": {
            "discovered": int(len(pairs)),
            "paired": int(pairs["paired"].sum()) if "paired" in pairs else 0,
            "ready": int(len(ready)),
        },
        "feature_rows": int(len(features)),
        "feature_count": int(
            sum(column not in {"subject_id", "n_ap_volumes", "n_fm_volumes"} for column in features)
        ),
        "clinical_n": clinical_n,
        "healthy_n": healthy_n,
        "analysis": analysis_status,
        "supplementary_figures": figures,
        "paper_figures": paper_figures,
        "method_differences": [
            "FSL preprocessing inputs: 100-s high-pass and 5-mm smoothing, rather than Tozzi's original preprocessing.",
            "SciPy Butterworth residual band-pass rather than FSL's exact temporal-filter implementation.",
            "AP/FM activations and PPI are labeled task adaptations; no Go/NoGo proxy is used.",
            "SPM-style PPI deconvolution/reconvolution remains missing; WM/CSF and six motion regressors are included.",
        ],
    }
    clinical_inputs = [Path(value) for value in config.get("inputs.clinical_files", [])]
    write_provenance(root / "provenance.json", config.path, clinical_inputs)
    report_dir = config.require_path("outputs.reports")
    report_path = _write_method_report(report_dir, status)
    status["report_html"] = str(report_path)
    status["report_pdf"] = str(report_dir / "report.pdf")
    (root / "run_summary.json").write_text(
        json.dumps(status, indent=2, default=str), encoding="utf-8"
    )
    return status


def analyze_tozzi_features(config: Config) -> dict[str, Any]:
    """Re-run clustering/reporting from cached task-residual features."""
    root = config.require_path("outputs.root")
    feature_path = root / "features" / "regional_features.tsv"
    participant_path = root / "cohort" / "participants.tsv"
    missing = [str(path) for path in (feature_path, participant_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cached Tozzi-method analysis inputs are missing: " + ", ".join(missing)
        )
    features = pd.read_csv(feature_path, sep="\t")
    participants = pd.read_csv(participant_path, sep="\t")
    with stage("Run clustering and statistical validation from cached features"):
        analysis = run_primary_analysis(config, features, participants)
    with stage("Generate Tozzi supplementary figures"):
        figures = _generate_supplementary_figures(config, root)
    with stage("Generate paper-style result figures"):
        paper_figures = _generate_paper_figures(config, analysis, root)
    pairs_path = root / "features" / "paired_inputs.tsv"
    pairs = pd.read_csv(pairs_path, sep="\t") if pairs_path.exists() else pd.DataFrame()
    status: dict[str, Any] = {
        "mode": "independent-tozzi-task-residual-cached-analysis",
        "task_pair_counts": {
            "discovered": int(len(pairs)),
            "paired": int(pairs["paired"].sum()) if "paired" in pairs else 0,
            "ready": int(pairs["ready"].sum()) if "ready" in pairs else 0,
        },
        "feature_rows": int(len(features)),
        "feature_count": int(
            sum(column not in {"subject_id", "n_ap_volumes", "n_fm_volumes"} for column in features)
        ),
        "clinical_n": int((participants.analysis_role == "clinical").sum()),
        "healthy_n": int((participants.analysis_role == "healthy_reference").sum()),
        "analysis": analysis,
        "supplementary_figures": figures,
        "paper_figures": paper_figures,
        "note": "Feature extraction was not rerun; analysis used cached regional_features.tsv.",
    }
    clinical_inputs = [Path(value) for value in config.get("inputs.clinical_files", [])]
    write_provenance(root / "provenance.json", config.path, clinical_inputs)
    report_path = _write_method_report(config.require_path("outputs.reports"), status)
    status["report_html"] = str(report_path)
    status["report_pdf"] = str(report_path.with_name("report.pdf"))
    (root / "run_summary.json").write_text(
        json.dumps(status, indent=2, default=str), encoding="utf-8"
    )
    return status
