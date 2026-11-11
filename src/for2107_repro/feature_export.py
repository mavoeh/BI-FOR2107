"""Export the extracted features that the stability analysis starts from.

The feature folder holds everything the analysis needs from feature
extraction, so it runs without the clinical database or the FEAT outputs:
the task (AP + FM) and rest-scan feature tables, the task feature catalog,
the eligible participants (role, diagnosis group, scanner, age, sex) and the
head motion of each task run. The feature tables are copied byte for byte.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, ConfigError
from .motion import task_motion_table
from .normative import read_normative_cohort
from .progress import report, stage
from .provenance import sha256, write_provenance

FILES = {
    "participants": "participants.tsv",
    "motion": "task_motion.tsv",
    "task_features": "task_features.tsv",
    "task_catalog": "task_feature_catalog.tsv",
    "rest_features": "rest_features.tsv",
}


def feature_files(root: Path) -> dict[str, Path]:
    """Paths of the exported tables; raises if any is missing."""
    paths = {name: root / filename for name, filename in FILES.items()}
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise ConfigError(
            f"{root} lacks {', '.join(missing)}; see features/README.md (export-features)"
        )
    return paths


def export_features(config: Config) -> dict[str, Any]:
    root = config.require_path("outputs.root")
    root.mkdir(parents=True, exist_ok=True)
    settings = config.get("feature_export", {}) or {}
    sources = {
        "task_features": Path(settings["task"]["feature_table"]),
        "task_catalog": Path(settings["task"]["catalog"]),
        "rest_features": Path(settings["rest"]["feature_table"]),
    }
    for name, source in sources.items():
        with stage(f"Copy {source.name} -> {FILES[name]}"):
            shutil.copyfile(source, root / FILES[name])

    with stage("Read eligible participants from the clinical database"):
        cohort, _ = read_normative_cohort(config, {})
    imaged = set()
    for name in ("task_features", "rest_features"):
        imaged |= set(pd.read_csv(sources[name], sep="\t", usecols=["subject_id"]).subject_id)
    participants = cohort.loc[cohort.subject_id.isin(imaged)].reset_index(drop=True)
    participants.to_csv(root / FILES["participants"], sep="\t", index=False)

    motion_settings = settings.get("motion", {}) or {}
    threshold = float(motion_settings.get("fd_threshold_mm", 0.3))
    with stage("Head motion of each task run"):
        motion = task_motion_table(
            config.require_path("inputs.task_data_root"),
            participants.subject_id,
            str(config.get("selection.session", "ses-01")),
            motion_settings.get("runs", ["AP", "FM"]),
            threshold,
        )
    motion.insert(1, "fd_threshold_mm", threshold)
    motion.to_csv(root / FILES["motion"], sep="\t", index=False)

    paths = feature_files(root)
    write_provenance(
        root / "provenance.json",
        config.path,
        [*sources.values(), *map(Path, config.get("inputs.clinical_files", []))],
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in paths.values()), encoding="utf-8"
    )
    counts = participants.groupby(["analysis_role", "diagnosis_group"]).size()
    report(f"Exported {len(participants)} participants to {root}")
    return {
        "root": str(root),
        "participants": {f"{role}/{group}": int(n) for (role, group), n in counts.items()},
        "files": {name: str(path) for name, path in paths.items()},
    }
