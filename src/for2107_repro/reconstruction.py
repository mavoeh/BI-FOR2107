"""Reconstruct Tozzi-like patient data from a population without subgroups.

A large pool is drawn from one smooth distribution (no subgroups) with the
correlations of Tozzi's 41 features: FOR2107 patient correlations within the
task-free and the task block, his published correlations between them
(Supplementary Figure 14a). Every pool point is labelled with the biotype
profile it correlates with most (his assignment rule), and the pool is then
reweighted as little as possible (minimum Kullback-Leibler divergence from
equal weights, i.e. entropy balancing) so that the labelled groups have his
sizes and per-feature mean profiles (Supplementary Figure 5) and the pool his
task-free x task correlations, each within its standard error.

Datasets of his size drawn with these weights are analysed with his pipeline
(silhouette and both null tests over k, leave-one-out and 20%-out stability,
split-half profile correlations, cluster sizes and profiles) and compared with
his published results. How far the weights had to move, and whether the
reweighted population is still a single cloud when sampled at large n,
measures how much subgroup structure his results require.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.metrics import adjusted_rand_score

from .clustering import fit_clusters
from .config import Config, ConfigError, load_config
from .progress import report, stage
from .provenance import write_provenance
from .replication import evaluate_structure
from .stability import (
    _template,
    assign_to_centroids,
    cluster_centroids,
    cosine_matrix,
    tozzi_correlation,
)

TASK_FREE_PREFIXES = ("D", "S", "A")


def is_task_free(code: str) -> bool:
    return code[0] in TASK_FREE_PREFIXES and code[1].isdigit()


# --- population without subgroups ---------------------------------------------------


def base_correlation(within: pd.DataFrame, cross: pd.DataFrame, codes: list[str]) -> np.ndarray:
    """41-feature correlation: FOR2107 within blocks, Tozzi's between blocks.

    `within` is the FOR2107 patient correlation of the available features
    (indexed by code; unavailable features are filled as in the stability
    analysis); `cross` holds his task x task-free values.
    """
    matrix = tozzi_correlation(within, codes)
    position = {code: index for index, code in enumerate(codes)}
    for task in cross.index:
        for edge in cross.columns:
            i, j = position[task], position[edge]
            matrix[i, j] = matrix[j, i] = float(cross.loc[task, edge])
    values, vectors = np.linalg.eigh(matrix)
    matrix = vectors @ np.diag(np.clip(values, 1e-3, None)) @ vectors.T
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


def draw_pool(
    size: int,
    mean: np.ndarray,
    correlation: np.ndarray,
    family: str,
    rng: np.random.Generator,
    df: float = 5.0,
) -> np.ndarray:
    """One smooth cloud with unit variances: normal or Student t."""
    factor = np.linalg.cholesky(correlation)
    pool = rng.standard_normal((size, len(mean))).astype(np.float32) @ factor.T.astype(np.float32)
    if family == "student_t":
        # scale mixture of normals, rescaled to unit variance
        chi = rng.chisquare(df, size=(size, 1)).astype(np.float32)
        pool *= np.sqrt((df - 2) / chi)
    elif family != "gaussian":
        raise ConfigError(f"Unknown pool family: {family}")
    return pool + mean.astype(np.float32)


def assign_profiles(pool: np.ndarray, profiles: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Index of the profile each point correlates with most (0-based)."""
    labels = np.empty(len(pool), dtype=np.int64)
    for start in range(0, len(pool), chunk):
        labels[start : start + chunk] = (
            assign_to_centroids(pool[start : start + chunk], profiles) - 1
        )
    return labels


# --- least-change reweighting ----------------------------------------------------


class EntropyBalance:
    """Weights closest to equal (KL) giving the labelled groups target sizes and
    mean profiles, and the pool target task-free x task correlations.

    Constraints are soft: each may miss its target by about its standard error
    (Gaussian penalty), which keeps the problem feasible and avoids chasing
    noise in the digitized targets. Solved in the dual with L-BFGS.
    """

    def __init__(
        self,
        pool: np.ndarray,
        labels: np.ndarray,
        sizes: np.ndarray,
        profiles: np.ndarray,
        size_tolerance: np.ndarray,
        profile_tolerance: np.ndarray,
        cross: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.pool = pool
        self.labels = labels
        self.sizes = sizes
        self.profiles = profiles
        self.k, self.p = profiles.shape
        self.size_var = size_tolerance**2
        self.profile_var = profile_tolerance**2
        # deviations from each point's own target profile
        self.deviation = pool - profiles[labels].astype(np.float32)
        self.cross = cross
        if cross is not None:
            rows, columns, targets, tolerance = cross
            centre = (sizes @ profiles).astype(np.float32)
            self.cross_left = pool[:, rows] - centre[rows]
            self.cross_right = pool[:, columns] - centre[columns]
            self.cross_target = targets
            self.cross_var = tolerance**2

    def _split(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        size = theta[: self.k]
        profile = theta[self.k : self.k + self.k * self.p].reshape(self.k, self.p)
        cross = theta[self.k + self.k * self.p :] if self.cross is not None else None
        if cross is not None:
            cross = cross.reshape(self.cross_left.shape[1], self.cross_right.shape[1])
        return size, profile, cross

    def scores(self, theta: np.ndarray) -> np.ndarray:
        size, profile, cross = self._split(theta)
        score = size[self.labels] + np.einsum(
            "ij,ij->i", self.deviation, profile[self.labels].astype(np.float32)
        )
        if cross is not None:
            score += np.einsum(
                "ij,ij->i", self.cross_left @ cross.astype(np.float32), self.cross_right
            )
        return score.astype(np.float64)

    def weights(self, theta: np.ndarray) -> np.ndarray:
        score = self.scores(theta)
        score -= score.max()
        weight = np.exp(score)
        return weight / weight.sum()

    def objective(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        size, profile, cross = self._split(theta)
        score = self.scores(theta)
        top = score.max()
        weight = np.exp(score - top)
        total = weight.sum()
        weight /= total
        value = top + np.log(total / len(score)) - size @ self.sizes
        value += 0.5 * (self.size_var @ size**2 + (self.profile_var * profile**2).sum())
        group = np.bincount(self.labels, weights=weight, minlength=self.k)
        grad_size = group - self.sizes + self.size_var * size
        grad_profile = np.zeros((self.k, self.p))
        for label in range(self.k):
            members = self.labels == label
            grad_profile[label] = weight[members] @ self.deviation[members]
        grad_profile += self.profile_var * profile
        gradient = [grad_size, grad_profile.ravel()]
        if cross is not None:
            moment = (self.cross_left * weight[:, None].astype(np.float32)).T @ self.cross_right
            value += -(cross * self.cross_target).sum() + 0.5 * (self.cross_var * cross**2).sum()
            gradient.append((moment - self.cross_target + self.cross_var * cross).ravel())
        return float(value), np.concatenate(gradient)

    def solve(self, max_iterations: int = 500) -> tuple[np.ndarray, dict[str, Any]]:
        dimension = self.k + self.k * self.p
        if self.cross is not None:
            dimension += self.cross_left.shape[1] * self.cross_right.shape[1]
        result = minimize(
            self.objective,
            np.zeros(dimension),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iterations, "maxcor": 30},
        )
        return self.weights(result.x), {
            "converged": bool(result.success),
            "iterations": int(result.nit),
            "message": str(result.message),
        }


def mixture_kl(
    means: np.ndarray,
    proportions: np.ndarray,
    centre: np.ndarray,
    correlation: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    within: np.ndarray | None = None,
) -> float:
    """KL divergence of real biotypes (a normal mixture with these means and
    within-biotype covariance) from one normal cloud with covariance
    `correlation`: the reweighting distance that genuine subgroups imply.

    `within` defaults to `correlation`; pass the total minus the
    between-biotype covariance to compare at an equal total. Monte Carlo over
    draws from the mixture.
    """
    within = correlation if within is None else within
    truth = rng.choice(len(proportions), size=draws, p=proportions)
    x = means[truth] + rng.standard_normal((draws, len(centre))) @ np.linalg.cholesky(within).T

    def log_density(mu: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        d = x - mu
        _, log_det = np.linalg.slogdet(covariance)
        return -0.5 * (np.einsum("ij,jk,ik->i", d, np.linalg.inv(covariance), d) + log_det)

    log_mixture = np.stack(
        [np.log(p) + log_density(mu, within) for p, mu in zip(proportions, means, strict=True)]
    )
    top = log_mixture.max(axis=0)
    log_mixture = top + np.log(np.exp(log_mixture - top).sum(axis=0))
    return float(np.mean(log_mixture - log_density(centre, correlation)))


def weight_diagnostics(weights: np.ndarray) -> dict[str, float]:
    """How far the weights moved from equal."""
    n = len(weights)
    positive = weights[weights > 0]
    return {
        "kl_from_equal": float(np.sum(positive * np.log(n * positive))),
        "effective_fraction": float(1.0 / np.sum(weights**2) / n),
        "max_weight_ratio": float(weights.max() * n),
        "share_weight_below_tenth": float(np.mean(weights < 0.1 / n)),
    }


def fitted_targets(
    pool: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    profiles: np.ndarray,
    profile_tolerance: np.ndarray,
) -> pd.DataFrame:
    """Weighted group sizes and the profile misfit in standard errors."""
    rows = []
    for label in range(profiles.shape[0]):
        members = labels == label
        share = weights[members].sum()
        mean = (weights[members] @ pool[members]) / share
        misfit = (mean - profiles[label]) / (profile_tolerance[label] / max(share, 1e-12))
        rows.append(
            {
                "biotype": label,
                "share": float(share),
                "mean_abs_misfit_se": float(np.mean(np.abs(misfit))),
                "max_abs_misfit_se": float(np.max(np.abs(misfit))),
                "profile_r": float(np.corrcoef(mean, profiles[label])[0, 1]),
            }
        )
    return pd.DataFrame(rows)


# --- Tozzi's pipeline on a dataset --------------------------------------------------


def tozzi_split_half(
    matrix: np.ndarray, k: int, iterations: int, rng: np.random.Generator
) -> float:
    """His Supp. Fig. 6 statistic: cluster half 1, assign half 2 by correlation,
    mean correlation of the cluster profiles between the halves."""
    values = []
    for _ in range(iterations):
        order = rng.permutation(len(matrix))
        half = len(matrix) // 2
        first, second = matrix[order[:half]], matrix[order[half : 2 * half]]
        centroids, _ = cluster_centroids(first, fit_clusters(first, k).labels, k)
        means, sizes = cluster_centroids(second, assign_to_centroids(second, centroids), k)
        r = [
            np.corrcoef(centroids[i], means[i])[0, 1]
            for i in range(k)
            if sizes[i] > 1 and np.isfinite(centroids[i]).all()
        ]
        values.append(np.mean(r))
    return float(np.median(values))


def tozzi_pipeline(
    matrix: np.ndarray,
    profiles: np.ndarray,
    truth: np.ndarray | None,
    settings: dict[str, Any],
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """His structure battery over k, plus the k = 6 solution's sizes, profile
    match to his biotypes and split-half reliability."""
    low, high = settings.get("k_range", [2, 15])
    k = int(settings.get("k", 6))
    structure = evaluate_structure(
        matrix, list(range(int(low), int(high) + 1)), settings["iterations"], seed, n_jobs
    )
    labels = fit_clusters(matrix, k).labels
    centroids, sizes = cluster_centroids(matrix, labels, k)
    similarity = cosine_matrix(centroids, profiles)
    rows, columns = linear_sum_assignment(-np.nan_to_num(similarity, nan=-2.0))
    order = np.argsort(columns)
    rng = np.random.default_rng(seed)
    summary = {
        "sizes_sorted": sorted(sizes.tolist(), reverse=True),
        "profile_cosine_mean": float(similarity[rows, columns].mean()),
        "profile_cosine_by_biotype": similarity[rows, columns][order].round(3).tolist(),
        "profile_mean_abs_diff": float(np.mean(np.abs(centroids[rows] - profiles[columns]))),
        "cluster_mean_abs_sd": float(np.mean(np.abs(centroids))),
        "split_half_r": tozzi_split_half(
            matrix, k, int(settings.get("split_half_iterations", 20)), rng
        ),
    }
    if truth is not None:
        summary["ari_to_assignment_labels"] = float(adjusted_rand_score(truth, labels))
    return structure, summary


# --- is the reweighted population still one cloud? ------------------------------------


def population_cluster_test(
    pool: np.ndarray,
    weights: np.ndarray | None,
    k: int,
    n: int,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Silhouette at large n against normal data with the same mean and
    covariance: a population with real subgroups exceeds it clearly."""
    from .clustering import _silhouette

    index = rng.choice(len(pool), size=n, p=weights)
    sample = pool[index].astype(np.float64)
    observed = _silhouette(fit_clusters(sample, k))
    mean, covariance = sample.mean(axis=0), np.cov(sample, rowvar=False)
    null = np.array(
        [
            _silhouette(fit_clusters(rng.multivariate_normal(mean, covariance, size=n), k))
            for _ in range(draws)
        ]
    )
    return {
        "large_n": n,
        "silhouette": float(observed),
        "normal_silhouette_mean": float(null.mean()),
        "normal_silhouette_sd": float(null.std(ddof=1)),
        "z_vs_normal": float((observed - null.mean()) / null.std(ddof=1)),
    }


def bimodality(pool: np.ndarray, labels: np.ndarray, weights: np.ndarray | None) -> float:
    """Largest bimodality coefficient along the directions separating the
    labelled groups (weighted LDA axes); above 0.555 suggests two modes."""
    w = np.full(len(pool), 1 / len(pool)) if weights is None else weights
    mean = w @ pool
    centred = pool - mean
    within = np.zeros((pool.shape[1], pool.shape[1]))
    between = np.zeros_like(within)
    for label in np.unique(labels):
        members = labels == label
        share = w[members].sum()
        group_mean = (w[members] @ pool[members]) / share
        diff = pool[members] - group_mean
        within += (diff * w[members, None]).T @ diff
        between += share * np.outer(group_mean - mean, group_mean - mean)
    values, vectors = eigh(between, within)
    coefficients = []
    for axis in vectors[:, np.argsort(values)[::-1][: len(np.unique(labels)) - 1]].T:
        projection = centred @ axis
        variance = w @ projection**2
        skew = (w @ projection**3) / variance**1.5
        kurtosis = (w @ projection**4) / variance**2
        coefficients.append((skew**2 + 1) / kurtosis)
    return float(max(coefficients))


# --- orchestration ------------------------------------------------------------------


def tozzi_structure(
    settings: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    """Planted settings, biotype template and proportions, and the 41-feature
    correlation (FOR2107 patients within blocks, Tozzi's between blocks)."""
    planted = load_config(settings["stability_config"]).get("stability.planted")
    template, proportions = _template({**planted, "level": "feature"})
    scores = pd.read_csv(settings["for2107_scores"], sep="\t")
    mapping = {name: code for name, code in planted["feature_codes"].items() if name in scores}
    patients = scores.loc[scores.analysis_role == "clinical", list(mapping)].rename(columns=mapping)
    cross = pd.read_csv(settings["correlations"], sep="\t", index_col="task")
    codes = list(template.columns)
    if settings.get("correlation_file"):
        # calibrated to his null silhouette curve (calibrate-tozzi)
        calibrated = pd.read_csv(settings["correlation_file"], sep="\t", index_col=0)
        return planted, template, proportions, calibrated.loc[codes, codes].to_numpy(dtype=float)
    return planted, template, proportions, base_correlation(patients.corr(), cross, codes)


def run_reconstruction(config: Config) -> dict[str, Any]:
    root = config.require_path("outputs.root")
    root.mkdir(parents=True, exist_ok=True)
    settings = config.get("reconstruction", {}) or {}
    seed = int(config.get("project.random_seed", 2107))
    n_jobs = int(config.get("analysis.n_jobs", 1))
    rng = np.random.default_rng(seed)
    planted, template, proportions, correlation = tozzi_structure(settings)
    codes = list(template.columns)
    profiles = template.to_numpy(dtype=float)
    sizes = np.array([planted["biotype_sizes"][name] for name in template.index], dtype=float)
    cross = pd.read_csv(settings["correlations"], sep="\t", index_col="task")
    pd.DataFrame(correlation, index=codes, columns=codes).to_csv(
        root / "base_correlation.tsv", sep="\t"
    )

    n = int(settings.get("n", 801))
    within_sd = float(settings.get("within_biotype_sd", 1.0))
    size_tolerance = np.full(len(sizes), 1e-3)
    profile_tolerance = (
        proportions[:, None] * within_sd / np.sqrt(sizes)[:, None] * np.ones_like(profiles)
    )
    rows = np.array([codes.index(edge) for edge in cross.columns])
    columns = np.array([codes.index(task) for task in cross.index])
    cross_target = cross.T.to_numpy(dtype=float)
    cross_tolerance = np.full_like(cross_target, 1 / np.sqrt(n))
    # his 6 biotype profiles sit around the pooled patient mean
    centre = proportions @ profiles

    # yardstick: how far real Tozzi biotypes (his planted mixture, same
    # within-biotype correlation) are from one cloud, in the same KL units
    scale = float(settings.get("reference_biotype_scale", 1.0))
    from .heavy_tails import within_biotype_correlation

    reference = {
        "biotype_scale": scale,
        "kl_real_biotypes": mixture_kl(
            scale * profiles, proportions, centre, correlation, 200_000, rng
        ),
        # the same biotypes with the total correlation held at the pool's
        "kl_real_biotypes_equal_total": mixture_kl(
            scale * profiles,
            proportions,
            centre,
            correlation,
            200_000,
            rng,
            within_biotype_correlation(correlation, scale * profiles, proportions),
        ),
    }
    report(f"KL of real Tozzi biotypes from one cloud: {reference['kl_real_biotypes']:.3f}")
    summaries: dict[str, Any] = {}
    structure_rows: list[pd.DataFrame] = []
    fit_rows: list[pd.DataFrame] = []
    for family, options in (settings.get("families", {"gaussian": {}}) or {}).items():
        with stage(f"{family}: draw a pool of {int(settings.get('pool_size', 1_000_000))} points"):
            pool = draw_pool(
                int(settings.get("pool_size", 1_000_000)),
                centre,
                correlation,
                family,
                rng,
                float((options or {}).get("df", 5)),
            )
            labels = assign_profiles(pool, profiles)
        report(
            f"{family}: share of pool per assigned profile {np.bincount(labels, minlength=6) / len(labels)}"
        )
        with stage(f"{family}: least-change reweighting"):
            balance = EntropyBalance(
                pool,
                labels,
                proportions,
                profiles,
                size_tolerance,
                profile_tolerance,
                (rows, columns, cross_target, cross_tolerance),
            )
            weights, solver = balance.solve(int(settings.get("max_iterations", 500)))
        diagnostics = weight_diagnostics(weights)
        report(f"{family}: {solver['message']}; {diagnostics}")
        # which targets demand the reweighting: sizes, + profiles, + correlations
        loose = np.full_like(profile_tolerance, 1e3)
        steps = {}
        with stage(f"{family}: reweighting with fewer targets"):
            for step, tolerance in (("sizes", loose), ("sizes_profiles", profile_tolerance)):
                partial, _ = EntropyBalance(
                    pool, labels, proportions, profiles, size_tolerance, tolerance, None
                ).solve(int(settings.get("max_iterations", 500)))
                steps[step] = weight_diagnostics(partial)
            steps["sizes_profiles_correlations"] = diagnostics
        fit = fitted_targets(pool, labels, weights, profiles, profile_tolerance).assign(
            family=family
        )
        fit["biotype"] = template.index[fit.biotype]
        fit_rows.append(fit)
        left, right = pool[:, rows] - centre[rows], pool[:, columns] - centre[columns]
        achieved = (left * weights[:, None]).T @ right
        family_summary: dict[str, Any] = {
            "solver": solver,
            "weights": diagnostics,
            "cross_correlation_r_with_tozzi": float(
                np.corrcoef(achieved.ravel(), cross_target.ravel())[0, 1]
            ),
            "cross_correlation_mean_abs_diff": float(np.mean(np.abs(achieved - cross_target))),
            "reweighting_by_targets": steps,
            "pool_share_by_profile": (np.bincount(labels, minlength=6) / len(labels))
            .round(3)
            .tolist(),
        }
        large = int(settings.get("population_n", 3000))
        with stage(f"{family}: is the population still one cloud?"):
            for name, w in (("unweighted", None), ("reweighted", weights)):
                family_summary[f"population_{name}"] = {
                    **population_cluster_test(
                        pool, w, 6, large, int(settings.get("population_draws", 20)), rng
                    ),
                    "max_bimodality_coefficient": bimodality(pool, labels, w),
                }
        for name, w in (("unweighted", None), ("reweighted", weights)):
            runs = []
            for dataset in range(int(settings.get("datasets", 6))):
                with stage(f"{family}, {name}: Tozzi's pipeline on dataset {dataset + 1}"):
                    index = rng.choice(len(pool), size=n, p=w)
                    structure, summary = tozzi_pipeline(
                        pool[index].astype(np.float64),
                        profiles,
                        labels[index],
                        settings,
                        int(rng.integers(0, 2**31 - 1)),
                        n_jobs,
                    )
                structure_rows.append(
                    structure.assign(family=family, weights=name, dataset=dataset)
                )
                runs.append(summary)
            family_summary[f"pipeline_{name}"] = runs
        summaries[family] = family_summary
        del pool, labels, weights, balance

    pd.concat(fit_rows).to_csv(root / "target_fit.tsv", sep="\t", index=False)
    pd.concat(structure_rows).to_csv(root / "pipeline_structure.tsv", sep="\t", index=False)
    write_provenance(
        root / "provenance.json",
        config.path,
        [
            Path(settings["for2107_scores"]),
            Path(settings["correlations"]),
            Path(planted["feature_means"]),
        ],
    )
    status = {
        "mode": "tozzi-reconstruction",
        "tozzi_published": settings.get("tozzi_published", {}),
        "reference": reference,
        "families": summaries,
    }
    (root / "status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    return status
