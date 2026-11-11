from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import run_primary_analysis
from .bids import stage_bids
from .clinical import write_cohort_tables
from .config import Config
from .derivatives import discover_preprocessed_runs, write_available_features
from .features import validate_feature_definitions
from .inventory import read_spss_metadata, write_inventory
from .preflight import write_preflight
from .progress import report, stage
from .provenance import write_provenance
from .report import render_report


def _add_cached_cohort_summary(summary: dict[str, Any], participants: pd.DataFrame) -> None:
    """Restore report cohort counts from the persisted participant table."""
    summary["cohort"] = {
        "clinical_with_imaging": int((participants["analysis_role"] == "clinical").sum()),
        "healthy_reference_with_imaging": int(
            (participants["analysis_role"] == "healthy_reference").sum()
        ),
    }


def rerender_report(config: Config) -> dict[str, str | None]:
    """Render the report from existing pipeline artifacts without recomputing features."""
    output_root = config.require_path("outputs.root")
    inventory = json.loads((output_root / "inventory" / "summary.json").read_text(encoding="utf-8"))
    participants = pd.read_csv(output_root / "cohort" / "participants.tsv", sep="\t")
    _add_cached_cohort_summary(inventory, participants)
    scans = pd.read_csv(output_root / "inventory" / "scans.tsv", sep="\t")
    features = pd.read_csv(output_root / "features" / "feature_validation.tsv", sep="\t")
    preflight = json.loads((output_root / "preflight.json").read_text(encoding="utf-8"))
    provenance = json.loads((output_root / "provenance.json").read_text(encoding="utf-8"))
    analysis = json.loads((output_root / "analysis" / "status.json").read_text(encoding="utf-8"))
    with stage("Render HTML and PDF report"):
        return render_report(
            config,
            inventory,
            scans,
            features,
            preflight,
            provenance,
            analysis,
        )


def run_cached_analysis_pipeline(config: Config) -> dict[str, Any]:
    """Rerun analysis and reporting from cached participant-level features."""
    output_root = config.require_path("outputs.root")
    feature_path = output_root / "features" / "regional_features.tsv"
    participant_path = output_root / "cohort" / "participants.tsv"
    required = [feature_path, participant_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Cached analysis inputs are missing: " + ", ".join(missing))

    with stage("Load cached participant-level features"):
        extracted_features = pd.read_csv(feature_path, sep="\t")
        participants = pd.read_csv(participant_path, sep="\t")
        report(
            f"Loaded {len(extracted_features)} cached participants; "
            "image feature extraction will not run"
        )
    with stage("Run clustering and statistical validation"):
        analysis = run_primary_analysis(config, extracted_features, participants)

    inventory_summary = json.loads(
        (output_root / "inventory" / "summary.json").read_text(encoding="utf-8")
    )
    _add_cached_cohort_summary(inventory_summary, participants)
    scans = pd.read_csv(output_root / "inventory" / "scans.tsv", sep="\t")
    feature_validation = pd.read_csv(output_root / "features" / "feature_validation.tsv", sep="\t")
    preflight = json.loads((output_root / "preflight.json").read_text(encoding="utf-8"))
    clinical_inputs = [Path(path) for path in config.get("inputs.clinical_files", [])]
    with stage("Record analysis provenance"):
        provenance = write_provenance(
            output_root / "provenance.json", config.path, [*clinical_inputs]
        )
    with stage("Render HTML and PDF report"):
        reports = render_report(
            config,
            inventory_summary,
            scans,
            feature_validation,
            preflight,
            provenance,
            analysis,
        )

    extraction_status_path = output_root / "features" / "extraction_status.json"
    extraction_status = json.loads(extraction_status_path.read_text(encoding="utf-8"))
    result = {
        "mode": "cached-feature-analysis",
        "cached_feature_table": str(feature_path),
        "inventory": inventory_summary,
        "feature_extraction": extraction_status,
        "analysis": analysis,
        "reports": reports,
    }
    (output_root / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report("Cached-feature analysis pipeline complete")
    return result


def _write_external_rest_inventory(config: Config) -> dict[str, Any]:
    runs = discover_preprocessed_runs(config)
    scans = pd.DataFrame(
        {
            "subject": runs.get("subject_id", pd.Series(dtype=str)),
            "session": runs.get("session", pd.Series(dtype=str)),
            "datatype": "func",
            "task": "rest",
            "suffix": "bold",
            "path": runs.get("bold", pd.Series(dtype=str)),
            "scanner": None,
        }
    )
    subjects = sorted(runs["subject_id"].unique()) if not runs.empty else []
    clinical_files = [Path(item) for item in config.get("inputs.clinical_files", [])]
    spss = []
    for path in clinical_files:
        item = read_spss_metadata(path)
        spss.append(
            {
                key: item.get(key)
                for key in ("path", "exists", "rows", "columns", "error")
                if key in item
            }
        )
    summary = {
        "mode": config.get("project.mode"),
        "session": config.get("selection.session"),
        "bids_root": str(config.require_path("inputs.preprocessed_rest_root")),
        "subject_count": len(subjects),
        "subjects": subjects,
        "scan_count": len(runs),
        "task_counts": {"rest": len(subjects)},
        "missing_task_events": 0,
        "has_top_level_dataset_description": False,
        "spss": spss,
        "limitations": [
            "External cleaned resting-state inputs; full preprocessing provenance is pending.",
            "High-motion-volume exclusion is disabled when framewise displacement is unavailable; affected subjects are retained as motion-QC unassessed.",
            "Task inputs are not included in this resting-state import.",
        ],
    }
    output = config.require_path("outputs.root") / "inventory"
    output.mkdir(parents=True, exist_ok=True)
    scans.to_csv(output / "scans.tsv", sep="\t", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "scans": scans}


def run_validation_pipeline(config: Config) -> dict[str, Any]:
    """Run the safe smoke path; scientific stages remain gated until inputs exist."""
    output_root = config.require_path("outputs.root")
    output_root.mkdir(parents=True, exist_ok=True)
    report("Pipeline started")
    with stage("1/9 Preflight checks"):
        preflight = write_preflight(config)
    external_rest = bool(config.get("inputs.preprocessed_rest_root"))
    with stage("2/9 Discover and inventory scans"):
        inventory = (
            _write_external_rest_inventory(config) if external_rest else write_inventory(config)
        )
        report(
            f"Discovered {inventory['summary']['scan_count']} scans from "
            f"{inventory['summary']['subject_count']} participants"
        )
    with stage("3/9 Select clinical and reference cohorts"):
        cohort = write_cohort_tables(
            config,
            set(inventory["summary"]["subjects"]),
            imaging_records=inventory["scans"],
        )
        report(
            f"Selected {cohort['summary']['clinical_with_imaging']} clinical and "
            f"{cohort['summary']['healthy_reference_with_imaging']} healthy-reference participants"
        )
    inventory["summary"]["cohort"] = cohort["summary"]
    with stage("4/9 Prepare analysis inputs"):
        staging = (
            {
                "skipped": True,
                "reason": "external-preprocessed-rest-input",
                "source_modified": False,
            }
            if external_rest
            else stage_bids(config)
        )
    with stage("5/9 Validate feature definitions"):
        features = validate_feature_definitions(config)
        features_path = output_root / "features" / "feature_validation.tsv"
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(features_path, sep="\t", index=False)
    with stage("6/9 Extract participant-level circuit features"):
        extracted = write_available_features(config)
    with stage("7/9 Run clustering and statistical validation"):
        if config.get("analysis.run_clustering", True):
            analysis = run_primary_analysis(config, extracted["features"], cohort["participants"])
        else:
            analysis = {"ready": False, "reasons": ["clustering-disabled-in-config"]}
    clinical_inputs = [Path(path) for path in config.get("inputs.clinical_files", [])]
    with stage("8/9 Record provenance"):
        provenance = write_provenance(
            output_root / "provenance.json", config.path, [*clinical_inputs]
        )
    with stage("9/9 Render HTML and PDF report"):
        reports = render_report(
            config,
            inventory["summary"],
            inventory["scans"],
            features,
            preflight,
            provenance,
            analysis,
        )
    result = {
        "preflight": preflight,
        "inventory": inventory["summary"],
        "cohort": cohort["summary"],
        "staging": staging,
        "feature_validation": str(features_path),
        "feature_extraction": extracted["status"],
        "analysis": analysis,
        "reports": reports,
    }
    (output_root / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report("Pipeline complete")
    return result
