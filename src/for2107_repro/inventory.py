from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import pandas as pd

from .config import Config

BIDS_IMAGE_RE = re.compile(
    r"(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)_bold\.nii(?:\.gz)?$"
)
SENSITIVE_JSON_KEYS = {
    "PatientName",
    "PatientBirthDate",
    "PatientID",
    "PatientSex",
    "PatientWeight",
    "AcquisitionDate",
    "AcquisitionDateTime",
    "InstitutionAddress",
}


@dataclass
class ScanRecord:
    subject: str
    session: str
    datatype: str
    task: str | None
    suffix: str
    path: str
    json_path: str | None
    events_path: str | None
    n_volumes: int | None
    repetition_time: float | None
    site: str | None
    scanner: str | None
    sensitive_keys: str


def _sidecar(path: Path) -> Path:
    name = path.name[:-7] + ".json" if path.name.endswith(".nii.gz") else path.with_suffix(".json").name
    return path.with_name(name)


def _events(path: Path) -> Path | None:
    match = BIDS_IMAGE_RE.match(path.name)
    if not match:
        return None
    return path.with_name(path.name.replace("_bold.nii.gz", "_events.tsv").replace("_bold.nii", "_events.tsv"))


def _scan_record(path: Path, bids_root: Path) -> ScanRecord:
    relative = path.relative_to(bids_root)
    subject = next((part for part in relative.parts if part.startswith("sub-")), "unknown")
    session = next((part for part in relative.parts if part.startswith("ses-")), "unknown")
    datatype = relative.parent.name
    match = BIDS_IMAGE_RE.match(path.name)
    task = match.group("task") if match else None
    suffix = path.name.split("_")[-1].replace(".nii.gz", "").replace(".nii", "")
    sidecar = _sidecar(path)
    metadata: dict[str, Any] = {}
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    shape = nib.load(str(path)).shape
    n_volumes = int(shape[3]) if len(shape) == 4 else None
    events = _events(path)
    sensitive = sorted(SENSITIVE_JSON_KEYS.intersection(metadata))
    return ScanRecord(
        subject=subject,
        session=session,
        datatype=datatype,
        task=task,
        suffix=suffix,
        path=str(path),
        json_path=str(sidecar) if sidecar.exists() else None,
        events_path=str(events) if events and events.exists() else None,
        n_volumes=n_volumes,
        repetition_time=metadata.get("RepetitionTime"),
        site=metadata.get("InstitutionName"),
        scanner=metadata.get("DeviceSerialNumber") or metadata.get("ManufacturersModelName"),
        sensitive_keys=",".join(sensitive),
    )


def scan_bids_like(bids_root: Path, session: str | None = None) -> pd.DataFrame:
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root does not exist: {bids_root}")
    paths = sorted(bids_root.glob("sub-*/ses-*/*/*.nii*"))
    records = [_scan_record(path, bids_root) for path in paths]
    frame = pd.DataFrame(asdict(record) for record in records)
    if frame.empty:
        return frame
    if session:
        frame = frame.loc[frame["session"] == session].reset_index(drop=True)
    return frame


def read_spss_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        result["error"] = "missing"
        return result
    try:
        import pyreadstat
    except ImportError:
        result["error"] = "pyreadstat-not-installed"
        return result
    try:
        _, metadata = pyreadstat.read_sav(path, metadataonly=True)
    except Exception as exc:  # pyreadstat exposes several parser-specific exceptions
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result.update(
        {
            "rows": metadata.number_rows,
            "columns": metadata.number_columns,
            "column_names": metadata.column_names,
            "column_labels": metadata.column_labels,
            "value_label_sets": sorted(metadata.value_labels),
        }
    )
    return result


def inventory(config: Config) -> dict[str, Any]:
    bids_root = config.require_path("inputs.bids_root")
    session = config.get("selection.session")
    scans = scan_bids_like(bids_root, session=session)
    clinical_files = [Path(item) for item in config.get("inputs.clinical_files", [])]
    subjects = sorted(scans["subject"].unique()) if not scans.empty else []
    task_counts = Counter(scans.loc[scans["datatype"] == "func", "task"].dropna()) if not scans.empty else Counter()
    missing_events = (
        scans.loc[(scans["datatype"] == "func") & (scans["task"] != "rest") & scans["events_path"].isna()]
        if not scans.empty
        else scans
    )
    spss_metadata = [read_spss_metadata(path) for path in clinical_files]
    spss_summary = [
        {key: item.get(key) for key in ("path", "exists", "rows", "columns", "error") if key in item}
        for item in spss_metadata
    ]
    summary = {
        "mode": config.get("project.mode"),
        "session": session,
        "bids_root": str(bids_root),
        "subject_count": len(subjects),
        "subjects": subjects,
        "scan_count": len(scans),
        "task_counts": dict(task_counts),
        "missing_task_events": len(missing_events),
        "has_top_level_dataset_description": (bids_root / "dataset_description.json").exists(),
        "spss": spss_summary,
        "limitations": [],
    }
    if config.get("project.mode") == "smoke":
        summary["limitations"].append("Ten-subject smoke test; no scientific cluster inference.")
    if len(missing_events):
        summary["limitations"].append("Task event files are missing; activation features are gated.")
    if not summary["has_top_level_dataset_description"]:
        summary["limitations"].append("Input is BIDS-like and requires non-destructive staging metadata.")
    return {"summary": summary, "scans": scans, "spss_metadata": spss_metadata}


def write_inventory(config: Config) -> dict[str, Any]:
    result = inventory(config)
    output = config.require_path("outputs.root") / "inventory"
    output.mkdir(parents=True, exist_ok=True)
    result["scans"].to_csv(output / "scans.tsv", sep="\t", index=False)
    summary = result["summary"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for index, item in enumerate(result["spss_metadata"], start=1):
        (output / f"spss_metadata_{index}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
    return result
