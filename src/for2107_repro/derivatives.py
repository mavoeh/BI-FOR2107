from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, ConfigError
from .features import extract_activation_feature, extract_rest_features, load_feature_definitions
from .progress import ProgressCounter, report


def _find_one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected one {description}, found {len(paths)}: {[str(path) for path in paths]}")
    return paths[0]


def discover_preprocessed_runs(config: Config) -> pd.DataFrame:
    external_root_value = config.get("inputs.preprocessed_rest_root")
    if external_root_value:
        return _discover_external_rest(config, Path(external_root_value))
    root = config.require_path("inputs.fmriprep_derivatives")
    session = config.get("selection.session")
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for bold in sorted(root.glob(f"sub-*/{session}/func/*_space-*_desc-preproc_bold.nii.gz")):
        name = bold.name
        subject = name.split("_")[0]
        task_part = next(part for part in name.split("_") if part.startswith("task-"))
        task = task_part.removeprefix("task-")
        prefix = name.split("_space-")[0]
        func_dir = bold.parent
        confounds = list(func_dir.glob(f"{prefix}_desc-confounds_timeseries.tsv"))
        metadata = bold.with_name(bold.name.replace(".nii.gz", ".json"))
        event = config.require_path("outputs.bids_staging") / subject / session / "func" / f"{prefix}_events.tsv"
        rows.append(
            {
                "subject_id": subject,
                "session": session,
                "task": task,
                "bold": str(bold),
                "confounds": str(confounds[0]) if len(confounds) == 1 else None,
                "metadata": str(metadata) if metadata.exists() else None,
                "events": str(event) if event.exists() else None,
                "ready": len(confounds) == 1 and metadata.exists() and (task == "rest" or event.exists()),
            }
        )
    return pd.DataFrame(rows)


def _discover_external_rest(config: Config, root: Path) -> pd.DataFrame:
    """Discover already-cleaned resting-state NIfTIs from an external workflow."""
    if not root.exists():
        return pd.DataFrame()
    session = config.get("selection.session")
    pattern = config.get(
        "inputs.preprocessed_rest_pattern",
        "sub-*_ses-*_filtered_func_data_clean_MNI.nii.gz",
    )
    rows = []
    for bold in sorted(root.rglob(pattern)):
        parts = bold.name.split("_")
        subject = next((part for part in parts if part.startswith("sub-")), None)
        file_session = next((part for part in parts if part.startswith("ses-")), None)
        if not subject or not file_session or (session and file_session != session):
            continue
        stem = bold.name.removesuffix(".nii.gz")
        metadata = bold.with_name(f"{stem}.json")
        bids_confounds = bold.with_name(
            f"{subject}_{file_session}_task-rest_desc-confounds_timeseries.tsv"
        )
        local_confounds = bold.with_name(f"{stem}_confounds.tsv")
        candidates = [path for path in (bids_confounds, local_confounds) if path.exists()]
        if not candidates:
            candidates = sorted(bold.parent.glob(f"{subject}_{file_session}*confounds*.tsv"))
        confounds = candidates[0] if len(candidates) == 1 else None
        rows.append(
            {
                "subject_id": subject,
                "session": file_session,
                "task": "rest",
                "bold": str(bold),
                "confounds": str(confounds) if confounds else None,
                "metadata": str(metadata) if metadata.exists() else None,
                "events": None,
                "ready": True,
                "source_format": "external_clean",
            }
        )
    return pd.DataFrame(rows)


def extract_available_features(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = discover_preprocessed_runs(config)
    if runs.empty:
        return pd.DataFrame(), pd.DataFrame(
            [{"stage": "feature_extraction", "ready": False, "reason": "no-fmriprep-derivatives"}]
        )
    definitions = load_feature_definitions(config)
    roi_root = Path(config.get("features.roi_directory"))
    def extract_subject(
        item: tuple[str, pd.DataFrame],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        subject, subject_runs = item
        values: dict[str, Any] = {"subject_id": subject}
        subject_audits: list[dict[str, Any]] = []
        rest = subject_runs.loc[subject_runs["task"] == "rest"]
        if len(rest) == 1 and bool(rest.iloc[0]["ready"]):
            row = rest.iloc[0]
            try:
                values.update(
                    extract_rest_features(
                        config,
                        subject,
                        Path(row["bold"]),
                        Path(row["confounds"]) if row["confounds"] else None,
                        Path(row["metadata"]) if row["metadata"] else None,
                        definitions,
                    )
                )
            except ConfigError as exc:
                subject_audits.append(
                    {
                        "subject_id": subject,
                        "task": "rest",
                        "ready": False,
                        "reason": "subject-excluded-rest-feature-extraction-failed",
                        "detail": str(exc),
                    }
                )
                return None, subject_audits
        else:
            subject_audits.append({"subject_id": subject, "task": "rest", "ready": False, "reason": "run-missing-or-ambiguous"})
        for feature in definitions:
            if feature.source_task == "rest":
                continue
            candidates = subject_runs.loc[subject_runs["task"] == feature.source_task]
            if not feature.ready or not feature.contrast:
                values[feature.name] = np.nan
                subject_audits.append(
                    {
                        "subject_id": subject,
                        "task": feature.source_task,
                        "feature": feature.name,
                        "ready": False,
                        "reason": "feature-mapping-not-final",
                    }
                )
                continue
            if len(candidates) != 1 or not bool(candidates.iloc[0]["ready"]):
                values[feature.name] = np.nan
                subject_audits.append(
                    {
                        "subject_id": subject,
                        "task": feature.source_task,
                        "feature": feature.name,
                        "ready": False,
                        "reason": "run-or-events-missing",
                    }
                )
                continue
            if feature.modality != "activation":
                values[feature.name] = np.nan
                subject_audits.append(
                    {
                        "subject_id": subject,
                        "task": feature.source_task,
                        "feature": feature.name,
                        "ready": False,
                        "reason": "task-connectivity-method-pending-event-documentation",
                    }
                )
                continue
            run = candidates.iloc[0]
            roi_paths = [roi_root / f"{name}.nii" for name in feature.rois]
            values[feature.name] = extract_activation_feature(
                Path(run["bold"]),
                Path(run["events"]),
                Path(run["confounds"]),
                Path(run["metadata"]),
                roi_paths,
                feature.contrast,
            )
        return values, subject_audits

    groups = list(runs.groupby("subject_id"))
    workers = max(1, int(config.get("features.extraction_jobs", 1)))
    report(f"Extracting features for {len(groups)} participants with {workers} workers")
    progress = ProgressCounter("Feature extraction", len(groups))
    extracted: list[
        tuple[dict[str, Any] | None, list[dict[str, Any]]] | None
    ] = [None] * len(groups)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_subject, group): index
            for index, group in enumerate(groups)
        }
        for future in as_completed(futures):
            result = future.result()
            extracted[futures[future]] = result
            values, subject_audits = result
            if values is None:
                subject = subject_audits[0]["subject_id"]
                report(f"Skipping {subject}: {subject_audits[0]['detail']}")
            progress.advance()
    completed = [item for item in extracted if item is not None]
    rows = [values for values, _ in completed if values is not None]
    audits = [audit for _, subject_audits in completed for audit in subject_audits]
    return pd.DataFrame(rows), pd.DataFrame(audits)


def write_available_features(config: Config) -> dict[str, Any]:
    features, audit = extract_available_features(config)
    output = config.require_path("outputs.root") / "features"
    output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / "regional_features.tsv", sep="\t", index=False)
    audit.to_csv(output / "extraction_audit.tsv", sep="\t", index=False)
    status = {
        "participants": len(features),
        "skipped_participants": int(
            audit.loc[
                audit.get("reason", pd.Series(index=audit.index, dtype=str))
                == "subject-excluded-rest-feature-extraction-failed",
                "subject_id",
            ].nunique()
            if "subject_id" in audit
            else 0
        ),
        "features": [column for column in features if column not in {"subject_id", "qc_pass", "high_motion_fraction"}],
        "audit_failures": int((audit.get("ready", pd.Series(dtype=bool)) == False).sum()),  # noqa: E712
    }
    (output / "extraction_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return {"status": status, "features": features, "audit": audit}
