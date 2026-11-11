"""Head-motion summaries from FSL MCFLIRT realignment parameters."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def framewise_displacement(parameters: np.ndarray, radius_mm: float = 50.0) -> np.ndarray:
    """Power et al. (2012) framewise displacement for volumes 2..n.

    MCFLIRT `.par` columns are three rotations (radians) followed by three
    translations (mm); rotations are converted to arc length on a sphere.
    """
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 2 or parameters.shape[1] < 6:
        raise ValueError("Expected an n x 6 realignment parameter matrix")
    deltas = np.abs(np.diff(parameters[:, :6], axis=0))
    return deltas[:, 3:].sum(axis=1) + radius_mm * deltas[:, :3].sum(axis=1)


def task_motion_table(
    task_root: Path,
    subjects: Iterable[str],
    session: str,
    tasks: Iterable[str],
    fd_threshold_mm: float,
) -> pd.DataFrame:
    """Per-participant mean FD and % high-motion volumes for each FEAT task run."""
    tasks = list(tasks)
    rows = []
    for subject in subjects:
        row: dict[str, float | str] = {"subject_id": subject}
        for task in tasks:
            path = (
                task_root
                / f"{task}_feat"
                / f"{subject}_{session}.feat"
                / "mc"
                / "prefiltered_func_data_mcf.par"
            )
            if path.exists():
                fd = framewise_displacement(np.loadtxt(path, ndmin=2))
                row[f"mean_fd_{task}"] = float(fd.mean())
                row[f"high_motion_pct_{task}"] = float(100 * np.mean(fd > fd_threshold_mm))
                row[f"n_frames_{task}"] = len(fd)
            else:
                row[f"mean_fd_{task}"] = np.nan
                row[f"high_motion_pct_{task}"] = np.nan
                row[f"n_frames_{task}"] = 0
        rows.append(row)
    table = pd.DataFrame(rows, columns=["subject_id"]) if not rows else pd.DataFrame(rows)
    table["mean_fd"] = table[[f"mean_fd_{task}" for task in tasks]].mean(axis=1, skipna=False)
    return table


def motion_exclusion_reasons(
    table: pd.DataFrame, tasks: Iterable[str], max_mean_fd_mm: float, max_high_motion_pct: float
) -> pd.Series:
    """First failing run criterion per participant; empty string when all runs pass."""
    reasons = pd.Series("", index=table.index)
    for task in tasks:
        checks = [
            (table[f"mean_fd_{task}"].isna(), f"motion-parameters-missing-{task}"),
            (
                table[f"mean_fd_{task}"] > max_mean_fd_mm,
                f"mean-fd-above-{max_mean_fd_mm:g}mm-{task}",
            ),
            (
                table[f"high_motion_pct_{task}"] > max_high_motion_pct,
                f"high-motion-volumes-above-{max_high_motion_pct:g}pct-{task}",
            ),
        ]
        for mask, reason in checks:
            reasons = reasons.where(~mask | (reasons != ""), reason)
    return reasons
