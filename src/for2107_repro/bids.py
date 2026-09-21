from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import Config
from .inventory import SENSITIVE_JSON_KEYS


def sanitized_metadata(source: Path) -> dict[str, Any]:
    metadata = json.loads(source.read_text(encoding="utf-8"))
    return {key: value for key, value in metadata.items() if key not in SENSITIVE_JSON_KEYS}


def _staged_relative(relative: Path) -> Path:
    replacements = {
        "_run-01_fieldmap": "_magnitude1",
        "_run-02_fieldmap": "_magnitude2",
        "_run-03_fieldmap": "_phasediff",
    }
    name = relative.name
    for old, new in replacements.items():
        name = name.replace(old, new)
    return relative.with_name(name)


def _staged_metadata(source: Path, target_relative: Path, source_root: Path) -> dict[str, Any]:
    metadata = sanitized_metadata(source)
    name = target_relative.name
    if "_task-" in name and "_bold.json" in name:
        task = next(part.removeprefix("task-") for part in name.split("_") if part.startswith("task-"))
        metadata["TaskName"] = task
    if name.endswith("_phasediff.json"):
        original = source.name
        echo1_path = source.with_name(original.replace("_run-03_fieldmap", "_run-01_fieldmap"))
        echo2_path = source.with_name(original.replace("_run-03_fieldmap", "_run-02_fieldmap"))
        echo1 = json.loads(echo1_path.read_text(encoding="utf-8"))["EchoTime"]
        echo2 = json.loads(echo2_path.read_text(encoding="utf-8"))["EchoTime"]
        if not float(echo1) < float(echo2):
            raise ValueError(f"Fieldmap echo times are inconsistent: {echo1_path}, {echo2_path}")
        metadata.pop("EchoNumber", None)
        metadata.pop("EchoTime", None)
        metadata["EchoTime1"] = float(echo1)
        metadata["EchoTime2"] = float(echo2)
        subject, session = target_relative.parts[:2]
        func_root = source_root / subject / session / "func"
        metadata["IntendedFor"] = [
            f"{session}/func/{path.name}" for path in sorted(func_root.glob("*_bold.nii.gz"))
        ]
    return metadata


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        return
    # The staging tree and source data share a filesystem. Hard links remain
    # resolvable inside a container, consume no duplicate image storage, and
    # are mounted read-only by the fMRIPrep wrapper.
    os.link(source, destination)


def stage_bids(config: Config) -> dict[str, Any]:
    source_root = config.require_path("inputs.bids_root")
    staging_root = config.require_path("outputs.bids_staging")
    session = config.get("selection.session")
    staging_root.mkdir(parents=True, exist_ok=True)
    description = {
        "Name": "FOR2107 non-destructive analysis staging dataset",
        "BIDSVersion": "1.10.0",
        "DatasetType": "raw",
        "Authors": ["FOR2107 consortium"],
        "GeneratedBy": [{"Name": "for2107-tozzi-repro", "Version": "0.1.0"}],
    }
    (staging_root / "dataset_description.json").write_text(
        json.dumps(description, indent=2), encoding="utf-8"
    )
    (staging_root / "README").write_text(
        "Non-destructive, deidentified staging dataset for the FOR2107 Tozzi adaptation.\n",
        encoding="utf-8",
    )
    stale_validation = staging_root / "staging_validation.json"
    if stale_validation.exists():
        stale_validation.unlink()
    for obsolete in staging_root.glob("sub-*/ses-*/fmap/*_run-*_fieldmap.*"):
        obsolete.unlink()
    counts = {"linked": 0, "sanitized_sidecars": 0, "events": 0}
    missing_events: list[str] = []
    for subject_dir in sorted(source_root.glob("sub-*")):
        for path in sorted(subject_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_root)
            if session and session not in relative.parts:
                continue
            target_relative = _staged_relative(relative)
            target = staging_root / target_relative
            if path.suffix == ".json":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(_staged_metadata(path, target_relative, source_root), indent=2),
                    encoding="utf-8",
                )
                counts["sanitized_sidecars"] += 1
            else:
                _link(path, target)
                counts["linked"] += 1
            if "_task-" in path.name and "_bold.nii" in path.name and "_task-rest_" not in path.name:
                event_name = path.name.replace("_bold.nii.gz", "_events.tsv").replace("_bold.nii", "_events.tsv")
                event_path = path.with_name(event_name)
                if event_path.exists():
                    counts["events"] += 1
                else:
                    missing_events.append(str(relative.with_name(event_name)))
    invalid_paths: list[str] = []
    try:
        from bids_validator import BIDSValidator

        validator = BIDSValidator()
        for path in staging_root.glob("sub-*/ses-*/*/*"):
            if path.is_file() or path.is_symlink():
                relative = "/" + str(path.relative_to(staging_root))
                if not validator.is_bids(relative):
                    invalid_paths.append(relative)
    except ImportError:
        invalid_paths.append("validation-unavailable:bids-validator-not-installed")
    validation = {
        **counts,
        "missing_task_events": sorted(set(missing_events)),
        "invalid_bids_paths": invalid_paths,
        "source_modified": False,
        "note": "Three-volume GRE fieldmaps were mapped to magnitude1, magnitude2, and phasediff using measured echo metadata; all session BOLD runs are IntendedFor targets.",
    }
    validation_path = config.require_path("outputs.root") / "inventory" / "staging_validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    return validation


def clear_broken_staging_links(staging_root: Path) -> int:
    """Remove only broken symlinks below an explicitly configured staging root."""
    removed = 0
    for path in staging_root.rglob("*"):
        if path.is_symlink() and not path.exists():
            os.unlink(path)
            removed += 1
    return removed
