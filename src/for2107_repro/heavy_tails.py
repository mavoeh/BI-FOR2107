"""Can patients without subgroups, but with heavier tails, give Tozzi's results?

Healthy controls are drawn from a multivariate normal with the correlations
of his 41 features; patients from the same structure with heavier tails, in
his sample sizes and studies (801 patients from four studies, 137 controls
from two). Each simulated sample is harmonized and scored like his (ComBat on
study, z-scores against the controls), clustered with his method, and run
through his validation battery over k = 2-15; the outcomes are compared with
his published results.

Tails that scale a whole profile (multivariate t) cannot matter: his
correlation distance ignores how extreme a person is overall. Feature-wise
tails (t marginals, outliers in single features) can, and so can
study-specific covariance, which survives ComBat's per-feature correction.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

from .config import Config
from .harmonize import combat_harmonize, healthy_reference_scores
from .progress import report, stage
from .reconstruction import tozzi_structure
from .stability import assign_to_centroids, cluster_centroids

DEFAULT_DESIGN = {
    "patients": {"RAD": 309, "HCP-DES": 208, "iSPOT-D": 198, "ENGAGE": 86},
    "controls": {"HCP-DES": 70, "iSPOT-D": 67},
}


# --- simulated samples --------------------------------------------------------------


def within_biotype_correlation(
    correlation: np.ndarray, means: np.ndarray, proportions: np.ndarray
) -> np.ndarray:
    """Covariance left within biotypes when the total equals `correlation`."""
    centred = means - proportions @ means
    within = correlation - (centred.T * proportions) @ centred
    values, vectors = np.linalg.eigh(within)
    return vectors @ np.diag(np.clip(values, 1e-3, None)) @ vectors.T


def patient_values(
    n: int,
    correlation: np.ndarray,
    scenario: dict[str, Any],
    rng: np.random.Generator,
    mean: np.ndarray | None = None,
    biotypes: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Patients: the controls' correlation structure with the scenario's tails.

    With `biotypes` (per-feature means, proportions) and a `biotypes` entry in
    the scenario, patients are drawn into Tozzi's biotypes at that strength;
    the within-biotype covariance keeps the total at `correlation`, so his
    multinormal null is unchanged. Otherwise they share the mean profile.
    """
    shift = None
    if "biotypes" in scenario and biotypes is not None:
        means, proportions = biotypes
        means = float(scenario["biotypes"]) * means
        correlation = within_biotype_correlation(correlation, means, proportions)
        shift = means[rng.choice(len(proportions), size=n, p=proportions)]
    factor = np.linalg.cholesky(correlation)
    z = rng.standard_normal((n, len(correlation))) @ factor.T
    if "whole_profile_t" in scenario:
        df = float(scenario["whole_profile_t"])
        z *= np.sqrt((df - 2) / rng.chisquare(df, size=(n, 1)))
    if "feature_t" in scenario:
        # Gaussian copula with t marginals, rescaled to unit variance
        df = float(scenario["feature_t"])
        z = stats.t.ppf(stats.norm.cdf(z), df) * np.sqrt((df - 2) / df)
    if "outliers" in scenario:
        rate, sd = float(scenario["outliers"]["rate"]), float(scenario["outliers"]["sd"])
        hit = rng.random(z.shape) < rate
        z[hit] += rng.normal(0, sd, hit.sum())
    if shift is not None:
        return z + shift
    return z if mean is None else z + mean


def simulate_sample(
    correlation: np.ndarray,
    scenario: dict[str, Any],
    design: dict[str, dict[str, int]],
    rng: np.random.Generator,
    mean: np.ndarray | None = None,
    biotypes: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Controls (normal) and patients (scenario tails) with study labels.

    A study factor adds study-specific covariance: within study s, every
    participant gets g * f_s with g ~ N(0, strength^2) and a random loading
    vector f_s, as different protocols would.
    """
    p = len(correlation)
    factor = np.linalg.cholesky(correlation)
    loadings = {
        study: rng.standard_normal(p) for study in {**design["patients"], **design["controls"]}
    }
    strength = float(scenario.get("study_factor", 0.0))
    frames = []
    for role, groups in (
        ("clinical", design["patients"]),
        ("healthy_reference", design["controls"]),
    ):
        for study, n in groups.items():
            if role == "clinical":
                values = patient_values(n, correlation, scenario, rng, mean, biotypes)
            else:
                values = rng.standard_normal((n, p)) @ factor.T
            if strength:
                values = values + strength * rng.standard_normal((n, 1)) * loadings[study]
            frame = pd.DataFrame(values, columns=[f"f{j}" for j in range(p)])
            frames.append(frame.assign(analysis_role=role, study=study))
    return pd.concat(frames, ignore_index=True)


def score_sample(sample: pd.DataFrame) -> np.ndarray:
    """His scoring: ComBat on study without covariates, z-scores against controls."""
    features = [column for column in sample if column.startswith("f")]
    harmonized = combat_harmonize(sample, features, "study")
    healthy = harmonized.analysis_role == "healthy_reference"
    scores, _ = healthy_reference_scores(harmonized, features, healthy)
    return scores.loc[~healthy, features].to_numpy(dtype=float)


# --- his validation battery, one tree per dataset -------------------------------------


def _tree(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = pdist(matrix, metric="correlation")
    return linkage(distances, method="average"), distances


def silhouette(square: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette from a square distance matrix (singletons score 0)."""
    return float(silhouette_values(square, labels).mean())


def silhouette_values(square: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Silhouette of every participant (singletons score 0)."""
    clusters, index = np.unique(labels, return_inverse=True)
    onehot = np.eye(len(clusters))[index]
    counts = onehot.sum(axis=0)
    sums = square @ onehot
    own = sums[np.arange(len(labels)), index]
    own_count = counts[index] - 1
    a = np.divide(own, own_count, out=np.zeros_like(own), where=own_count > 0)
    mean_other = sums / counts
    mean_other[np.arange(len(labels)), index] = np.inf
    b = mean_other.min(axis=1)
    return np.where(own_count > 0, (b - a) / np.maximum(a, b), 0.0)


def _silhouettes(matrix: np.ndarray, ks: list[int]) -> np.ndarray:
    tree, distances = _tree(matrix)
    square = squareform(distances)
    return np.array([silhouette(square, fcluster(tree, k, criterion="maxclust")) for k in ks])


def battery(
    matrix: np.ndarray,
    ks: list[int],
    settings: dict[str, Any],
    rng: np.random.Generator,
    studies: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Silhouette with multinormal and permutation nulls, leave-one-out and
    20%-out stability over k (as in his 05_clustering script; p is his plain
    exceedance fraction), and the k = 6 solution's sizes, profile size and
    split-half profile correlation."""
    tree, distances = _tree(matrix)
    square = squareform(distances)
    labels = {k: fcluster(tree, k, criterion="maxclust") for k in ks}
    observed = np.array([silhouette(square, labels[k]) for k in ks])
    mean, covariance = matrix.mean(axis=0), np.cov(matrix, rowvar=False)
    normal = np.array(
        [
            _silhouettes(rng.multivariate_normal(mean, covariance, size=len(matrix)), ks)
            for _ in range(int(settings.get("simulations", 200)))
        ]
    )
    permuted = np.array(
        [
            _silhouettes(np.column_stack([rng.permutation(column) for column in matrix.T]), ks)
            for _ in range(int(settings.get("permutations", 100)))
        ]
    )
    loo = []
    for left_out in rng.choice(
        len(matrix), size=int(settings.get("leave_one_out", 60)), replace=False
    ):
        keep = np.arange(len(matrix)) != left_out
        refit, _ = _tree(matrix[keep])
        loo.append(
            [adjusted_rand_score(labels[k][keep], fcluster(refit, k, "maxclust")) for k in ks]
        )
    subsample = []
    size = round(0.8 * len(matrix))
    for _ in range(int(settings.get("subsample", 40))):
        keep = rng.choice(len(matrix), size=size, replace=False)
        refit, _ = _tree(matrix[keep])
        subsample.append(
            [adjusted_rand_score(labels[k][keep], fcluster(refit, k, "maxclust")) for k in ks]
        )
    by_k = pd.DataFrame(
        {
            "k": ks,
            "silhouette": observed,
            "p_multinormal": (normal > observed).mean(axis=0),
            "p_permuted": (permuted > observed).mean(axis=0),
            "loo_ari": np.mean(loo, axis=0),
            "subsample_ari": np.mean(subsample, axis=0),
        }
    )
    k6 = int(settings.get("k", 6))
    centroids, sizes = cluster_centroids(matrix, labels[k6], k6)
    split_half = []
    for _ in range(int(settings.get("split_half", 10))):
        order = rng.permutation(len(matrix))
        first, second = matrix[order[: len(matrix) // 2]], matrix[order[len(matrix) // 2 :]]
        half_tree, _ = _tree(first)
        half_centroids, _ = cluster_centroids(first, fcluster(half_tree, k6, "maxclust"), k6)
        means, counts = cluster_centroids(second, assign_to_centroids(second, half_centroids), k6)
        split_half.append(
            np.mean(
                [
                    np.corrcoef(half_centroids[i], means[i])[0, 1]
                    for i in range(k6)
                    if counts[i] > 1 and np.isfinite(half_centroids[i]).all()
                ]
            )
        )
    shares = np.sort(sizes)[::-1] / len(matrix)
    widths = silhouette_values(square, labels[k6])
    by_size = np.argsort(-sizes)
    summary = {
        "share_negative_silhouette": float(np.mean(widths < 0)),
        # association of the k = 6 clusters with study (his Supp. Table 11: 161.4)
        "study_chi2": float(stats.chi2_contingency(pd.crosstab(labels[k6], studies))[0])
        if studies is not None
        else float("nan"),
        "cluster_silhouettes_by_size": [
            round(float(widths[labels[k6] == cluster + 1].mean()), 3)
            for cluster in by_size
            if sizes[cluster]
        ],
        "sizes_sorted": np.sort(sizes)[::-1].tolist(),
        "largest_four_min_share": float(shares[:4].min()),
        "smallest_two_share": float(shares[4:].sum()),
        "profile_mean_abs": float(np.nanmean(np.abs(centroids))),
        "split_half_r": float(np.median(split_half)),
        "share_beyond_3sd": float(np.mean(np.abs(matrix) > 3)),
        "share_beyond_4sd": float(np.mean(np.abs(matrix) > 4)),
    }
    return by_k, summary


def tozzi_like(
    by_k: pd.DataFrame, summary: dict[str, Any], criteria: dict[str, Any]
) -> dict[str, bool]:
    """Each published property of his solution, and whether this dataset has it."""
    at = by_k.set_index("k")
    k = int(criteria.get("k", 6))
    from_k = int(criteria.get("significant_from_k", 5))
    low, high = criteria.get("silhouette_k6_range", [0.045, 0.085])
    profile_low, profile_high = criteria.get("profile_abs_range", [0.2, 0.4])
    checks = {
        "p_multinormal_k6": at.loc[k, "p_multinormal"] < 0.05,
        "p_multinormal_all_k_from_5": bool((at.loc[from_k:, "p_multinormal"] < 0.05).all()),
        "p_permuted_k6": at.loc[k, "p_permuted"] < 0.05,
        "loo_all_k": bool((at.loo_ari > criteria.get("loo_above", 0.75)).all()),
        "subsample_all_k": bool((at.subsample_ari > criteria.get("subsample_above", 0.28)).all()),
        "silhouette_k6": low <= at.loc[k, "silhouette"] <= high,
        "size_pattern": summary["largest_four_min_share"]
        >= criteria.get("largest_four_share", 0.12)
        and summary["smallest_two_share"] <= criteria.get("smallest_two_share", 0.10),
        "split_half": summary["split_half_r"] >= criteria.get("split_half_above", 0.85),
        "profile_size": profile_low <= summary["profile_mean_abs"] <= profile_high,
        "negative_silhouettes": abs(
            summary["share_negative_silhouette"] - criteria.get("share_negative", 0.285)
        )
        <= criteria.get("share_negative_tolerance", 0.07),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    checks["all"] = all(checks.values())
    return checks


def _single_thread() -> None:
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def _dataset_task(task: tuple[Any, ...]) -> tuple[pd.DataFrame, dict[str, Any]]:
    name, scenario, replicate, correlation, design, settings, criteria, seed, mean, biotypes = task
    rng = np.random.default_rng(seed)
    sample = simulate_sample(correlation, scenario, design, rng, mean, biotypes)
    studies = sample.loc[sample.analysis_role == "clinical", "study"].to_numpy()
    matrix = score_sample(sample)
    low, high = settings.get("k_range", [2, 15])
    by_k, summary = battery(matrix, list(range(int(low), int(high) + 1)), settings, rng, studies)
    row = {
        "scenario": name,
        "replicate": replicate,
        **summary,
        **tozzi_like(by_k, summary, criteria),
    }
    return by_k.assign(scenario=name, replicate=replicate), row


def run_heavy_tails(config: Config) -> dict[str, Any]:
    root = config.require_path("outputs.root")
    root.mkdir(parents=True, exist_ok=True)
    settings = config.get("heavy_tails", {}) or {}
    seed = int(config.get("project.random_seed", 2107))
    n_jobs = int(config.get("analysis.n_jobs", 1))
    _, template, proportions, correlation = tozzi_structure(settings)
    profiles = template.to_numpy(dtype=float)
    # patients' mean profile: the size-weighted mean of his biotype profiles
    mean = proportions @ profiles if settings.get("patient_mean") == "tozzi" else None
    design = settings.get("design", DEFAULT_DESIGN)
    criteria = settings.get("tozzi_like", {}) or {}
    battery_settings = settings.get("battery", {}) or {}
    seeds = iter(np.random.SeedSequence(seed).generate_state(100_000))
    tasks = [
        (
            name,
            scenario or {},
            replicate,
            correlation,
            design,
            battery_settings,
            criteria,
            next(seeds),
            mean,
            (profiles, proportions),
        )
        for name, scenario in settings["scenarios"].items()
        for replicate in range(int(settings.get("replicates", 20)))
    ]
    with stage(f"Simulate and analyse {len(tasks)} samples of Tozzi's design"):
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs, initializer=_single_thread) as executor:
                results = list(executor.map(_dataset_task, tasks))
        else:
            results = [_dataset_task(task) for task in tasks]
    by_k = pd.concat([frame for frame, _ in results], ignore_index=True)
    datasets = pd.DataFrame([row for _, row in results])
    by_k.to_csv(root / "battery_by_k.tsv", sep="\t", index=False)
    datasets.to_csv(root / "datasets.tsv", sep="\t", index=False)
    checks = [column for column in datasets if datasets[column].dtype == bool]
    summary = datasets.groupby("scenario", sort=False)[checks].mean()
    medians = (
        by_k.loc[by_k.k == int(criteria.get("k", 6))]
        .groupby("scenario", sort=False)[
            ["silhouette", "p_multinormal", "p_permuted", "loo_ari", "subsample_ari"]
        ]
        .median()
    )
    tails = datasets.groupby("scenario", sort=False)[
        [
            "share_beyond_3sd",
            "share_beyond_4sd",
            "split_half_r",
            "profile_mean_abs",
            "share_negative_silhouette",
            "study_chi2",
        ]
    ].median()
    summary = summary.join(medians.add_suffix("_k6_median")).join(tails.add_suffix("_median"))
    summary.to_csv(root / "summary.tsv", sep="\t")
    report(
        "Share of samples with each of Tozzi's published properties:\n"
        + summary[checks].round(2).to_string()
    )
    status = {
        "mode": "heavy-tail-hypothesis",
        "replicates": int(settings.get("replicates", 20)),
        "design": design,
        "tozzi_published": settings.get("tozzi_published", {}),
        "summary": summary.reset_index().to_dict(orient="records"),
    }
    (root / "status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    return status
