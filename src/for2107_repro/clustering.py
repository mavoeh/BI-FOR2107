from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .progress import ProgressCounter, report


class ClusteringError(ValueError):
    pass


def _worker_count(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    return max(1, n_jobs)


def _batches(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _batch_size(total: int, workers: int) -> int:
    return max(1, (total + workers * 2 - 1) // (workers * 2))


def _bounded_process_map(
    function: Callable[[Any], list[float]],
    tasks: Iterable[Any],
    *,
    executor: ProcessPoolExecutor,
    workers: int,
    progress: ProgressCounter,
) -> list[float]:
    """Process batches in input order with bounded parent-process memory."""
    results: list[float] = []
    pending = deque()
    capacity = max(2, workers * 2)
    for task in tasks:
        pending.append(executor.submit(function, task))
        if len(pending) >= capacity:
            completed = pending.popleft().result()
            results.extend(completed)
            progress.advance(len(completed))
    while pending:
        completed = pending.popleft().result()
        results.extend(completed)
        progress.advance(len(completed))
    return results


def _silhouette_batch(task: tuple[list[np.ndarray], int]) -> list[float]:
    matrices, candidate = task
    return [_silhouette(fit_clusters(matrix, candidate)) for matrix in matrices]


def _loo_batch(task: tuple[np.ndarray, np.ndarray, int, list[int]]) -> list[float]:
    matrix, original, candidate, omitted_values = task
    scores = []
    for omitted in omitted_values:
        keep = np.arange(len(matrix)) != omitted
        scores.append(
            adjusted_rand_score(original[keep], fit_clusters(matrix[keep], candidate).labels)
        )
    return scores


def _subsample_batch(
    task: tuple[np.ndarray, np.ndarray, int, list[np.ndarray]],
) -> list[float]:
    matrix, original, candidate, index_values = task
    return [
        adjusted_rand_score(original[indices], fit_clusters(matrix[indices], candidate).labels)
        for indices in index_values
    ]


def _split_half_batch(
    task: tuple[np.ndarray, int, list[tuple[int, int]]],
) -> list[dict[str, float | int]]:
    matrix, n_clusters, iterations = task
    rows: list[dict[str, float | int]] = []
    for iteration, seed in iterations:
        result = split_half_replication(matrix, n_clusters, seed=seed)
        correlations = np.asarray(result["profile_correlations"], dtype=float)
        for cluster, correlation in enumerate(correlations, start=1):
            rows.append(
                {
                    "iteration": iteration,
                    "cluster": cluster,
                    "profile_correlation": float(correlation),
                    "mean_profile_correlation": float(result["mean_profile_correlation"]),
                }
            )
    return rows


@dataclass(frozen=True)
class ClusterFit:
    labels: np.ndarray
    linkage_matrix: np.ndarray
    condensed_distance: np.ndarray

    @property
    def distance_matrix(self) -> np.ndarray:
        return squareform(self.condensed_distance)


def _matrix(values: pd.DataFrame | np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ClusteringError("Clustering requires at least 3 participants and 2 features")
    if not np.isfinite(matrix).all():
        raise ClusteringError("Clustering input contains missing or infinite values")
    if np.any(np.std(matrix, axis=1) == 0):
        raise ClusteringError("Correlation distance is undefined for constant participant profiles")
    return matrix


def correlation_distance(values: pd.DataFrame | np.ndarray) -> np.ndarray:
    return pdist(_matrix(values), metric="correlation")


def fit_clusters(values: pd.DataFrame | np.ndarray, n_clusters: int) -> ClusterFit:
    matrix = _matrix(values)
    if not 2 <= n_clusters < len(matrix):
        raise ClusteringError(f"n_clusters must be between 2 and {len(matrix) - 1}")
    distances = correlation_distance(matrix)
    hierarchy = linkage(distances, method="average")
    labels = fcluster(hierarchy, n_clusters, criterion="maxclust")
    return ClusterFit(labels, hierarchy, distances)


def within_cluster_distance_sum(fit: ClusterFit) -> float:
    distances = fit.distance_matrix
    total = 0.0
    for cluster in np.unique(fit.labels):
        indices = np.flatnonzero(fit.labels == cluster)
        local = distances[np.ix_(indices, indices)]
        total += float(local[np.tril_indices_from(local, k=-1)].sum())
    return total


def _silhouette(fit: ClusterFit) -> float:
    return float(silhouette_score(fit.distance_matrix, fit.labels, metric="precomputed"))


def empirical_p_value(exceedances: int, trials: int) -> float:
    """Return a finite-sample-corrected Monte Carlo p-value."""
    if trials < 1 or not 0 <= exceedances <= trials:
        raise ValueError("Empirical p-values require 0 <= exceedances <= trials and trials >= 1")
    return float((exceedances + 1) / (trials + 1))


def evaluate_solutions(
    values: pd.DataFrame | np.ndarray, candidates: Iterable[int]
) -> pd.DataFrame:
    matrix = _matrix(values)
    candidates = list(candidates)
    progress = ProgressCounter("Candidate cluster solutions", len(candidates))
    rows = []
    for candidate in candidates:
        if not 2 <= candidate < len(matrix):
            progress.advance()
            continue
        fit = fit_clusters(matrix, candidate)
        rows.append(
            {
                "n_clusters": candidate,
                "within_cluster_distance": within_cluster_distance_sum(fit),
                "silhouette": _silhouette(fit),
            }
        )
        progress.advance()
    return pd.DataFrame(rows)


def null_silhouette_tests(
    values: pd.DataFrame | np.ndarray,
    candidates: Iterable[int],
    *,
    simulations: int,
    permutations: int,
    seed: int = 2107,
    n_jobs: int = 1,
) -> pd.DataFrame:
    matrix = _matrix(values)
    candidates = list(candidates)
    rng = np.random.default_rng(seed)
    mean = matrix.mean(axis=0)
    covariance = np.cov(matrix, rowvar=False)
    valid_candidates = [candidate for candidate in candidates if 2 <= candidate < len(matrix)]
    total = len(valid_candidates) * (simulations + permutations)
    progress = ProgressCounter("Silhouette null tests", total)
    workers = _worker_count(n_jobs)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    rows = []
    try:
        for candidate in candidates:
            if not 2 <= candidate < len(matrix):
                continue
            report(
                f"Testing k={candidate}: {simulations} simulations + {permutations} permutations"
            )
            observed = _silhouette(fit_clusters(matrix, candidate))
            simulated_inputs = (
                rng.multivariate_normal(mean, covariance, size=len(matrix), check_valid="ignore")
                for _ in range(simulations)
            )
            if executor is None:
                simulated_scores = [
                    _silhouette(fit_clusters(values, candidate)) for values in simulated_inputs
                ]
                progress.advance(simulations)
            else:
                size = _batch_size(simulations, workers)
                tasks = ((batch, candidate) for batch in _batches(simulated_inputs, size))
                simulated_scores = _bounded_process_map(
                    _silhouette_batch,
                    tasks,
                    executor=executor,
                    workers=workers,
                    progress=progress,
                )
            permuted_inputs = (
                np.column_stack(
                    [rng.permutation(matrix[:, column]) for column in range(matrix.shape[1])]
                )
                for _ in range(permutations)
            )
            if executor is None:
                permuted_scores = [
                    _silhouette(fit_clusters(values, candidate)) for values in permuted_inputs
                ]
                progress.advance(permutations)
            else:
                size = _batch_size(permutations, workers)
                tasks = ((batch, candidate) for batch in _batches(permuted_inputs, size))
                permuted_scores = _bounded_process_map(
                    _silhouette_batch,
                    tasks,
                    executor=executor,
                    workers=workers,
                    progress=progress,
                )
            simulated_exceedances = int(np.sum(np.asarray(simulated_scores) >= observed))
            permuted_exceedances = int(np.sum(np.asarray(permuted_scores) >= observed))
            rows.append(
                {
                    "n_clusters": candidate,
                    "silhouette": observed,
                    "p_multinormal": empirical_p_value(simulated_exceedances, simulations),
                    "p_permuted": empirical_p_value(permuted_exceedances, permutations),
                    "multinormal_exceedances": simulated_exceedances,
                    "permutation_exceedances": permuted_exceedances,
                    "simulations": simulations,
                    "permutations": permutations,
                }
            )
    finally:
        if executor is not None:
            executor.shutdown()
    return pd.DataFrame(rows)


def leave_one_out_stability(
    values: pd.DataFrame | np.ndarray,
    candidates: Iterable[int],
    *,
    n_jobs: int = 1,
) -> pd.DataFrame:
    matrix = _matrix(values)
    candidates = list(candidates)
    valid_candidates = [candidate for candidate in candidates if 2 <= candidate < len(matrix) - 1]
    progress = ProgressCounter("Leave-one-out stability", len(valid_candidates) * len(matrix))
    workers = _worker_count(n_jobs)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    rows = []
    try:
        for candidate in candidates:
            if not 2 <= candidate < len(matrix) - 1:
                continue
            original = fit_clusters(matrix, candidate).labels
            if executor is None:
                scores = _loo_batch((matrix, original, candidate, list(range(len(matrix)))))
                progress.advance(len(matrix))
            else:
                size = _batch_size(len(matrix), workers)
                tasks = (
                    (matrix, original, candidate, batch)
                    for batch in _batches(range(len(matrix)), size)
                )
                scores = _bounded_process_map(
                    _loo_batch,
                    tasks,
                    executor=executor,
                    workers=workers,
                    progress=progress,
                )
            rows.append(
                {
                    "n_clusters": candidate,
                    "loo_ari_mean": np.mean(scores),
                    "loo_ari_sd": np.std(scores, ddof=1),
                }
            )
    finally:
        if executor is not None:
            executor.shutdown()
    return pd.DataFrame(rows)


def subsample_stability(
    values: pd.DataFrame | np.ndarray,
    candidates: Iterable[int],
    *,
    fraction: float = 0.8,
    iterations: int = 10000,
    seed: int = 2107,
    n_jobs: int = 1,
) -> pd.DataFrame:
    matrix = _matrix(values)
    candidates = list(candidates)
    rng = np.random.default_rng(seed)
    size = max(3, int(round(len(matrix) * fraction)))
    valid_candidates = [candidate for candidate in candidates if 2 <= candidate < size]
    progress = ProgressCounter("Subsample stability", len(valid_candidates) * iterations)
    workers = _worker_count(n_jobs)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    rows = []
    try:
        for candidate in candidates:
            if not 2 <= candidate < size:
                continue
            original = fit_clusters(matrix, candidate).labels
            indices_iter = (
                np.sort(rng.choice(len(matrix), size=size, replace=False))
                for _ in range(iterations)
            )
            if executor is None:
                index_values = list(indices_iter)
                scores = _subsample_batch((matrix, original, candidate, index_values))
                progress.advance(iterations)
            else:
                batch_size = _batch_size(iterations, workers)
                tasks = (
                    (matrix, original, candidate, batch)
                    for batch in _batches(indices_iter, batch_size)
                )
                scores = _bounded_process_map(
                    _subsample_batch,
                    tasks,
                    executor=executor,
                    workers=workers,
                    progress=progress,
                )
            rows.append(
                {
                    "n_clusters": candidate,
                    "subsample_ari_mean": np.mean(scores),
                    "subsample_ari_sd": np.std(scores, ddof=1),
                    "iterations": iterations,
                    "fraction": fraction,
                }
            )
    finally:
        if executor is not None:
            executor.shutdown()
    return pd.DataFrame(rows)


def assign_by_profile_correlation(values: np.ndarray, cluster_profiles: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    cluster_profiles = np.asarray(cluster_profiles, dtype=float)
    assignments = []
    for row in values:
        correlations = [np.corrcoef(row, profile)[0, 1] for profile in cluster_profiles]
        assignments.append(int(np.nanargmax(correlations)) + 1)
    return np.asarray(assignments)


def split_half_replication(
    values: pd.DataFrame | np.ndarray, n_clusters: int, *, seed: int = 2107
) -> dict[str, np.ndarray | float]:
    matrix = _matrix(values)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(matrix))
    first_indices, second_indices = np.array_split(shuffled, 2)
    first_fit = fit_clusters(matrix[first_indices], n_clusters)
    profiles = np.vstack(
        [
            matrix[first_indices][first_fit.labels == cluster].mean(axis=0)
            for cluster in range(1, n_clusters + 1)
        ]
    )
    second_labels = assign_by_profile_correlation(matrix[second_indices], profiles)
    second_profiles = np.vstack(
        [
            matrix[second_indices][second_labels == cluster].mean(axis=0)
            if np.any(second_labels == cluster)
            else np.full(matrix.shape[1], np.nan)
            for cluster in range(1, n_clusters + 1)
        ]
    )
    profile_correlations = np.asarray(
        [np.corrcoef(profiles[index], second_profiles[index])[0, 1] for index in range(n_clusters)]
    )
    return {
        "first_indices": first_indices,
        "second_indices": second_indices,
        "first_labels": first_fit.labels,
        "second_labels": second_labels,
        "profile_correlations": profile_correlations,
        "mean_profile_correlation": float(np.nanmean(profile_correlations)),
    }


def repeated_split_half_replication(
    values: pd.DataFrame | np.ndarray,
    n_clusters: int,
    *,
    iterations: int,
    seed: int = 2107,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Repeat profile-correlation split-half replication deterministically."""
    matrix = _matrix(values)
    if iterations < 1:
        raise ValueError("split-half iterations must be positive")
    rng = np.random.default_rng(seed)
    iteration_seeds = [
        (iteration, int(value))
        for iteration, value in enumerate(rng.integers(0, 2**31 - 1, size=iterations), start=1)
    ]
    workers = _worker_count(n_jobs)
    progress = ProgressCounter("Repeated split-half replication", iterations)
    if workers == 1:
        batches = [iteration_seeds]
        rows = _split_half_batch((matrix, n_clusters, iteration_seeds))
        progress.advance(iterations)
    else:
        size = _batch_size(iterations, workers)
        batches = list(_batches(iteration_seeds, size))
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for batch, result in zip(
                batches,
                executor.map(
                    _split_half_batch,
                    ((matrix, n_clusters, batch) for batch in batches),
                ),
                strict=True,
            ):
                rows.extend(result)
                progress.advance(len(batch))
    return pd.DataFrame(rows).sort_values(["iteration", "cluster"]).reset_index(drop=True)
