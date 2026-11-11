"""Calibrate the correlation structure of Tozzi's features to his own null test.

His multinormal null draws patients from a normal distribution with his
data's mean and covariance, so each published pair (mean silhouette at k,
p-value at k; Supplementary Figure 2a/b) says where his observed silhouette
lies in the null distribution for that k. Only his task-free x task
correlations are published; the correlations within the task-free and the
task block are taken from FOR2107 and their dominant patterns strengthened
by one parameter gamma (eigenvalues raised to gamma, unit diagonal kept)
until the null reproduces his pairs: observed_k = median_k + sd_k * z(1 - p_k)
in least squares over k = 2-15. gamma = 1 is the FOR2107 structure.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config
from .heavy_tails import _silhouettes
from .progress import report, stage
from .reconstruction import tozzi_structure

KS = list(range(2, 16))


def task_free_mask(codes: list[str]) -> np.ndarray:
    return np.array([code[0] in "DSA" and code[1].isdigit() for code in codes])


def sharpen(correlation: np.ndarray, gamma: float, task_free: np.ndarray) -> np.ndarray:
    """Strengthen the dominant patterns within each block; keep the cross block."""
    result = correlation.copy()
    for block in (task_free, ~task_free):
        values, vectors = np.linalg.eigh(correlation[np.ix_(block, block)])
        sharpened = vectors @ np.diag(np.clip(values, 1e-6, None) ** gamma) @ vectors.T
        scale = np.sqrt(np.diag(sharpened))
        result[np.ix_(block, block)] = sharpened / np.outer(scale, scale)
    values, vectors = np.linalg.eigh(result)
    result = vectors @ np.diag(np.clip(values, 1e-3, None)) @ vectors.T
    scale = np.sqrt(np.diag(result))
    return result / np.outer(scale, scale)


def _null_draw(task: tuple[np.ndarray, np.ndarray, int, int]) -> np.ndarray:
    mean, correlation, n, seed = task
    rng = np.random.default_rng(seed)
    return _silhouettes(rng.multivariate_normal(mean, correlation, size=n), KS)


def null_curves(
    mean: np.ndarray, correlation: np.ndarray, n: int, draws: int, seed: int, n_jobs: int
) -> np.ndarray:
    """Mean silhouette at k = 2-15 for `draws` normal samples (draws x 14)."""
    seeds = np.random.SeedSequence(seed).generate_state(draws)
    tasks = [(mean, correlation, n, int(s)) for s in seeds]
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        return np.array(list(executor.map(_null_draw, tasks)))


def misfit(curves: np.ndarray, observed: pd.DataFrame) -> float:
    """Squared distance of his pairs from this null, in null SD units."""
    median = np.median(curves, axis=0)
    sd = curves.std(axis=0, ddof=1)
    expected_z = stats.norm.ppf(1 - observed.p_multinormal.clip(0.0005, 0.9995).to_numpy())
    z = (observed.silhouette.to_numpy() - median) / sd
    return float(np.sum((z - expected_z) ** 2))


def run_calibration(config: Config) -> dict[str, Any]:
    root = config.require_path("outputs.root")
    root.mkdir(parents=True, exist_ok=True)
    settings = config.get("calibration", {}) or {}
    seed = int(config.get("project.random_seed", 2107))
    n_jobs = int(config.get("analysis.n_jobs", 1))
    _, template, proportions, base = tozzi_structure(settings)
    codes = list(template.columns)
    task_free = task_free_mask(codes)
    centre = proportions @ template.to_numpy(dtype=float)
    observed = pd.read_csv(settings["curves"], sep="\t").set_index("k").loc[KS]
    n = int(settings.get("n", 801))
    draws = int(settings.get("draws", 100))
    rows, curves_by_gamma = [], {}
    for gamma in settings.get("gammas", [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]):
        with stage(f"Null silhouette curves at gamma {gamma}"):
            curves = null_curves(
                centre, sharpen(base, float(gamma), task_free), n, draws, seed, n_jobs
            )
        curves_by_gamma[float(gamma)] = curves
        rows.append({"gamma": float(gamma), "misfit": misfit(curves, observed)})
        report(f"gamma {gamma}: misfit {rows[-1]['misfit']:.2f}")
    table = pd.DataFrame(rows)
    # refine on a parabola through the three best grid points
    best = table.nsmallest(3, "misfit").sort_values("gamma")
    a, b, _ = np.polyfit(best.gamma, best.misfit, 2)
    gamma = (
        float(np.clip(-b / (2 * a), best.gamma.min(), best.gamma.max()))
        if a > 0
        else float(best.gamma.iloc[1])
    )
    calibrated = sharpen(base, gamma, task_free)
    with stage(f"Null silhouette curves at calibrated gamma {gamma:.2f}"):
        curves = null_curves(centre, calibrated, n, draws, seed + 1, n_jobs)
    table.to_csv(root / "gamma_misfit.tsv", sep="\t", index=False)
    pd.DataFrame(calibrated, index=codes, columns=codes).to_csv(
        root / "calibrated_correlation.tsv", sep="\t"
    )
    median, sd = np.median(curves, axis=0), curves.std(axis=0, ddof=1)
    fit = observed.assign(
        null_median=median,
        null_sd=sd,
        null_p=[(curves[:, i] > observed.silhouette.iloc[i]).mean() for i in range(len(KS))],
        excess_over_null=observed.silhouette.to_numpy() - median,
    )
    fit.to_csv(root / "null_fit.tsv", sep="\t")
    eigen = {
        name: (np.sort(np.linalg.eigvalsh(matrix[np.ix_(mask, mask)]))[::-1][:3] / mask.sum())
        .round(3)
        .tolist()
        for name, mask in (("task_free", task_free), ("task", ~task_free))
        for matrix in (calibrated,)
    }
    status = {
        "mode": "tozzi-null-calibration",
        "gamma": gamma,
        "misfit_calibrated": misfit(curves, observed),
        "share_of_variance_top3_patterns": eigen,
        "grid": table.to_dict(orient="records"),
    }
    (root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status
