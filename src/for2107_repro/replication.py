"""Tozzi-faithful replication analyses for FOR2107.

Mirrors the published analysis code (leotozzi88/cluster_study_2023,
analysis_main) as closely as FOR2107 allows: per-scan motion exclusion (>25%
of frames with FD > 0.3 mm), random-forest imputation within scanner, ComBat
on scanner without covariates, healthy-control z-scores, PPI directions
averaged after standardization, correlation-distance average-linkage
clustering, and the same structure, stability and clinical-generalization
tests. Results are set against Tozzi's own numbers for the matching feature
subset where those were published. Two further analyses bound what FOR2107
can show: agreement of profiles and cluster assignments across independent
measurements, and the power to detect Tozzi-strength biotypes.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

from .clustering import (
    assign_by_profile_correlation,
    evaluate_solutions,
    fit_clusters,
    leave_one_out_stability,
    null_silhouette_tests,
    subsample_stability,
)
from .config import Config
from .harmonize import combat_harmonize, healthy_reference_scores, random_forest_pmm
from .motion import task_motion_table
from .normative import read_normative_cohort
from .progress import report, stage
from .provenance import write_provenance

CIRCUIT_ORDER = [
    "D_C",
    "S_C",
    "A_C",
    "NS_A",
    "NS_C",
    "NTC_A",
    "NTC_C",
    "NTN_A",
    "NTN_C",
    "P_A",
    "C_A",
    "C_C",
]


def circuit_of(feature: str) -> str:
    """Tozzi circuit-modality label (e.g. ``D_C``, ``NS_A``) of a feature name."""
    if feature.endswith("_connectivity"):
        for key, label in (("_default_", "D_C"), ("_salience_", "S_C"), ("_attention_", "A_C")):
            if key in feature:
                return label
    modality = "C" if "_ppi" in feature else "A" if "_activation" in feature else None
    for prefix, label in (("Negativesad", "NS"), ("Negativethreat", "NTC"), ("Positive", "P")):
        if modality and f"_{prefix}_" in feature:
            return f"{label}_{modality}"
    raise ValueError(f"Cannot map feature to a Tozzi circuit: {feature}")


def select_features(columns: list[str], groups: dict[str, str]) -> dict[str, list[str]]:
    """Feature columns per group, from one regular expression per group."""
    selected = {
        group: [column for column in columns if re.search(pattern, column)]
        for group, pattern in groups.items()
    }
    empty = [group for group, names in selected.items() if not names]
    if empty:
        raise ValueError(f"Feature groups without matching columns: {empty}")
    return selected


def mask_high_motion(
    data: pd.DataFrame,
    groups: dict[str, list[str]],
    motion_runs: dict[str, list[str]],
    motion: pd.DataFrame,
    max_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set a scan's features to missing when too many of its frames moved.

    For each group, the high-motion fraction is pooled over the runs it was
    derived from (both task runs for the concatenated task-free series), as
    Tozzi did per scan. Participants without motion parameters are kept.
    """
    result = data.copy()
    audit = pd.DataFrame({"subject_id": data["subject_id"]})
    merged = data[["subject_id"]].merge(motion, on="subject_id", how="left")
    for group, runs in motion_runs.items():
        moved = sum(
            merged[f"high_motion_pct_{run}"].fillna(0) / 100 * merged[f"n_frames_{run}"].fillna(0)
            for run in runs
        )
        frames = sum(merged[f"n_frames_{run}"].fillna(0) for run in runs)
        fraction = (moved / frames.where(frames > 0)).to_numpy()
        flagged = np.nan_to_num(fraction, nan=0.0) > max_fraction
        result.loc[flagged, groups[group]] = np.nan
        audit[f"high_motion_fraction_{group}"] = fraction
        audit[f"excluded_{group}"] = flagged
    return result, audit


def faithful_scores(
    data: pd.DataFrame,
    features: list[str],
    batch_column: str,
    healthy_mask: pd.Series,
    *,
    seed: int,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Impute within scanner, ComBat without covariates, z-score to controls.

    Features named ``<base>__phys_<roi>`` are the two PPI directions; as in
    Tozzi's code they are standardized separately and then averaged.
    """
    imputed, audit = random_forest_pmm(data, features, batch_column, seed=seed, n_jobs=n_jobs)
    harmonized = combat_harmonize(imputed, features, batch_column)
    scores, _ = healthy_reference_scores(harmonized, features, healthy_mask)
    final: list[str] = []
    for feature in features:
        base = feature.split("__phys_")[0]
        if base == feature:
            final.append(feature)
        elif base not in final:
            directions = [name for name in features if name.split("__phys_")[0] == base]
            scores[base] = scores[directions].mean(axis=1)
            final.append(base)
    return scores, final, audit


def tozzi_name(circuit_means: pd.Series, threshold: float = 0.5) -> str:
    """Tozzi-style label: circuits whose mean deviates by at least 0.5 SD."""
    parts = [
        f"{circuit}{'+' if value > 0 else '-'}"
        for circuit, value in circuit_means.items()
        if abs(value) >= threshold
    ]
    return " ".join(parts) or "none >= 0.5 SD"


def cluster_profiles(scores: pd.DataFrame, labels: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Cluster sizes, circuit-level mean profiles and Tozzi-style names."""
    circuits = pd.Series({feature: circuit_of(feature) for feature in features})
    rows = []
    for cluster in np.unique(labels):
        members = scores.loc[labels == cluster, features]
        circuit_means = members.mean().groupby(circuits).mean()
        circuit_means = circuit_means.reindex([c for c in CIRCUIT_ORDER if c in circuit_means])
        rows.append(
            {
                "cluster": int(cluster),
                "n": len(members),
                **circuit_means.round(3).to_dict(),
                "tozzi_style_name": tozzi_name(circuit_means),
            }
        )
    return pd.DataFrame(rows)


def evaluate_structure(
    matrix: np.ndarray, candidates: list[int], iterations: dict[str, int], seed: int, n_jobs: int
) -> pd.DataFrame:
    """Elbow, silhouette nulls and stability, as in Tozzi's 05_clustering script."""
    table = evaluate_solutions(matrix, candidates)[["n_clusters", "within_cluster_distance"]]
    nulls = null_silhouette_tests(
        matrix,
        candidates,
        simulations=int(iterations["simulations"]),
        permutations=int(iterations["permutations"]),
        seed=seed,
        n_jobs=n_jobs,
    )
    # Tozzi's p-value is the plain exceedance fraction (no +1 correction).
    nulls["p_multinormal_tozzi"] = nulls.multinormal_exceedances / nulls.simulations
    nulls["p_permuted_tozzi"] = nulls.permutation_exceedances / nulls.permutations
    loo = leave_one_out_stability(matrix, candidates, n_jobs=n_jobs)
    subsample = subsample_stability(
        matrix,
        candidates,
        fraction=0.8,
        iterations=int(iterations["subsample"]),
        seed=seed,
        n_jobs=n_jobs,
    )
    return (
        table.merge(nulls, on="n_clusters")
        .merge(loo[["n_clusters", "loo_ari_mean"]], on="n_clusters")
        .merge(subsample[["n_clusters", "subsample_ari_mean"]], on="n_clusters")
    )


def benchmark(structure: pd.DataFrame, reference: dict[int, dict[str, float]]) -> pd.DataFrame:
    """FOR2107 against Tozzi's published values at the same k."""
    rows = []
    for k, values in reference.items():
        observed = structure.loc[structure.n_clusters == int(k)]
        if observed.empty:
            continue
        for metric, tozzi_value in values.items():
            column = {"p_multinormal": "p_multinormal_tozzi", "loo_ari": "loo_ari_mean"}.get(
                metric, "subsample_ari_mean" if metric == "subsample_ari" else metric
            )
            rows.append(
                {
                    "n_clusters": int(k),
                    "metric": metric,
                    "tozzi": float(tozzi_value),
                    "for2107": float(observed.iloc[0][column]),
                }
            )
    return pd.DataFrame(rows)


def _centroids(matrix: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clusters = np.unique(labels)
    return clusters, np.vstack([matrix[labels == cluster].mean(axis=0) for cluster in clusters])


def match_labels(
    reference: np.ndarray, reference_labels: np.ndarray, matrix: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Relabel `labels` to the reference clusters with the most similar centroids."""
    ref_ids, ref_centroids = _centroids(reference, reference_labels)
    ids, centroids = _centroids(matrix, labels)
    similarity = np.corrcoef(centroids, ref_centroids)[: len(ids), len(ids) :]
    rows, columns = linear_sum_assignment(-similarity)
    mapping = {ids[row]: ref_ids[column] for row, column in zip(rows, columns, strict=True)}
    return np.array([mapping.get(label, -1) for label in labels])


CLINICAL_TESTS = ("mann_whitney", "tozzi_wilcoxon")


def compare_clusters(
    values: pd.DataFrame, labels: np.ndarray, measures: list[str], test: str = "mann_whitney"
) -> pd.DataFrame:
    """Compare each cluster with the other clinical participants.

    ``mann_whitney``: two-sample Mann-Whitney U test (as described in Tozzi's
    methods); r is the rank-biserial correlation. ``tozzi_wilcoxon``: the test
    in Tozzi's compare_clusters code, a one-sample Wilcoxon of the cluster
    against the other participants' median with r = z / sqrt(N). The latter
    tests symmetry about that median, so it rejects for skewed or
    zero-inflated scales even when the groups do not differ.
    """
    if test not in CLINICAL_TESTS:
        raise ValueError(f"Unknown clinical test: {test}")
    rows = []
    for cluster in np.unique(labels[labels > 0]):
        for measure in measures:
            inside = values.loc[labels == cluster, measure].dropna()
            outside = values.loc[labels != cluster, measure].dropna()
            if len(inside) < 5 or outside.empty:
                continue
            reference = float(outside.median())
            if test == "mann_whitney":
                result = stats.mannwhitneyu(inside, outside)
                r = float(2 * result.statistic / (len(inside) * len(outside)) - 1)
                z = float(np.sign(r) * stats.norm.isf(result.pvalue / 2))
            else:
                differences = inside - reference
                if not differences.any():
                    continue
                result = stats.wilcoxon(differences, zero_method="wilcox", method="approx")
                sign = np.sign(differences.median() or differences.mean())
                z = float(sign * abs(result.zstatistic))
                r = z / np.sqrt(len(inside))
            rows.append(
                {
                    "cluster": int(cluster),
                    "measure": measure,
                    "n": len(inside),
                    "median": float(inside.median()),
                    "median_other": reference,
                    "z": z,
                    "r": r,
                    "p_value": float(result.pvalue),
                }
            )
    return pd.DataFrame(rows)


def clinical_test_calibration(
    values: pd.DataFrame,
    labels: np.ndarray,
    measures: list[str],
    seed: int,
    iterations: int = 200,
) -> pd.DataFrame:
    """False-positive rate of each clinical test under shuffled cluster labels."""
    rng = np.random.default_rng(seed)
    rows = []
    for test in CLINICAL_TESTS:
        tables = [
            compare_clusters(values, rng.permutation(labels), measures, test)
            for _ in range(iterations)
        ]
        stacked = pd.concat(tables)
        for measure, group in stacked.groupby("measure"):
            rows.append(
                {
                    "test": test,
                    "measure": measure,
                    "false_positive_rate": float((group.p_value < 0.05).mean()),
                    "pct_positive_direction": float(100 * (group.z > 0).mean()),
                    "skewness": float(stats.skew(values[measure].dropna())),
                }
            )
    return pd.DataFrame(rows)


def _one_sided(table: pd.DataFrame, direction: pd.Series) -> pd.Series:
    same = np.sign(table["z"].to_numpy()) == direction.to_numpy()
    return pd.Series(
        np.where(same, table["p_value"] / 2, 1 - table["p_value"] / 2), index=table.index
    )


def clinical_generalization(
    matrix: np.ndarray,
    labels: np.ndarray,
    cohort: pd.DataFrame,
    measures: list[str],
    k: int,
    seed: int,
    method: str = "mann_whitney",
) -> pd.DataFrame:
    """Whole-sample, split-half and leave-site-out cluster comparisons.

    A difference is generalizable when it is significant in the whole sample
    and in both parts of a split (the held-out part tested one-sided in the
    whole-sample direction), as in Tozzi's split-half and leave-study-out
    procedures. Split clusters are matched to whole-sample clusters by
    centroid correlation. `method` selects the test (see compare_clusters).
    """
    values = cohort.reset_index(drop=True)
    whole = compare_clusters(values, labels, measures, method).rename(
        columns={"p_value": "p_whole", "z": "z_whole", "r": "r_whole"}
    )
    key = ["cluster", "measure"]

    def held_out_test(train: np.ndarray, test: np.ndarray, name: str) -> pd.DataFrame:
        train_labels = match_labels(
            matrix, labels, matrix[train], fit_clusters(matrix[train], k).labels
        )
        ids, centroids = _centroids(matrix[train], train_labels)
        test_labels = ids[assign_by_profile_correlation(matrix[test], centroids) - 1]
        train_table = compare_clusters(values.iloc[train], train_labels, measures, method)
        test_table = compare_clusters(values.iloc[test], test_labels, measures, method)
        merged = (
            whole[[*key, "z_whole"]]
            .merge(train_table, on=key)
            .merge(test_table, on=key, suffixes=("_train", "_test"))
        )
        direction = np.sign(merged["z_whole"])
        # derivation part: two-sided, but only counts in the whole-sample direction
        merged[f"p_{name}_a"] = np.where(
            np.sign(merged["z_train"]) == direction, merged["p_value_train"], 1.0
        )
        merged[f"p_{name}_b"] = _one_sided(
            merged.rename(columns={"z_test": "z", "p_value_test": "p_value"}), direction
        )
        return merged[[*key, f"p_{name}_a", f"p_{name}_b"]]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(matrix))
    half = len(matrix) // 2
    table = whole.merge(held_out_test(order[:half], order[half:], "split_half"), on=key, how="left")
    for site in sorted(values["site"].dropna().unique()):
        test = np.flatnonzero(values["site"].to_numpy() == site)
        train = np.flatnonzero(values["site"].to_numpy() != site)
        table = table.merge(held_out_test(train, test, f"held_out_{site}"), on=key, how="left")
    significant = table["p_whole"] < 0.05
    table["generalizable_split_half"] = (
        significant & (table["p_split_half_a"] < 0.05) & (table["p_split_half_b"] < 0.05)
    )
    held = [
        column for column in table if column.startswith("p_held_out_") and column.endswith("_b")
    ]
    table["generalizable_leave_site_out"] = significant & (table[held] < 0.05).all(axis=1)
    return table


def _profile_scores(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    edges = []
    for feature in a.columns:
        edges.append(
            {
                "feature": feature,
                "pearson_r": float(np.corrcoef(a[feature], b[feature])[0, 1]),
                "icc_a1": _icc_a1(a[feature].to_numpy(), b[feature].to_numpy()),
            }
        )
    return pd.DataFrame(edges)


def _icc_a1(first: np.ndarray, second: np.ndarray) -> float:
    """ICC(A,1): two-way, absolute agreement, single measurement."""
    values = np.column_stack([first, second])
    n, k = values.shape
    grand = values.mean()
    rows = values.mean(axis=1)
    columns = values.mean(axis=0)
    ms_rows = k * np.sum((rows - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((columns - grand) ** 2) / (k - 1)
    residual = values - rows[:, None] - columns[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    return float(
        (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n)
    )


def measurement_agreement(
    a: pd.DataFrame, b: pd.DataFrame, ks: list[int], seed: int, permutations: int = 200
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Agreement of two measurements of the same profiles in the same people.

    Returns per-feature agreement, cluster-assignment agreement per k, and a
    summary. Assignment uses Tozzi's transfer rule (nearest centroid by
    profile correlation) for both measurements: cluster measurement A, then
    assign each person's A profile and B profile to the A centroids. Their
    agreement is the test-retest reliability of biotype membership; the
    centroid rule's agreement with the hierarchical labels is its ceiling.
    """
    features = list(a.columns)
    first, second = a[features].to_numpy(), b[features].to_numpy()
    profile_r = np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(first, second, strict=True)])
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        labels_a = fit_clusters(first, k).labels
        labels_b = fit_clusters(second, k).labels
        _, centroids = _centroids(first, labels_a)
        assigned_a = assign_by_profile_correlation(first, centroids)
        assigned_b = assign_by_profile_correlation(second, centroids)
        chance = [
            cohen_kappa_score(assigned_a, assigned_b[rng.permutation(len(assigned_b))])
            for _ in range(permutations)
        ]
        rows.append(
            {
                "n_clusters": k,
                "n": len(first),
                "ari_independent_clusterings": adjusted_rand_score(labels_a, labels_b),
                "kappa_retest": cohen_kappa_score(assigned_a, assigned_b),
                "kappa_retest_chance_95pct": float(np.percentile(chance, 95)),
                "pct_same_biotype": float(100 * np.mean(assigned_a == assigned_b)),
                "ari_centroid_rule_vs_hierarchical": adjusted_rand_score(
                    labels_a, np.unique(labels_a)[assigned_a - 1]
                ),
            }
        )
    summary = {
        "n": len(first),
        "median_profile_correlation": float(np.median(profile_r)),
        "profile_correlation_iqr": [float(v) for v in np.percentile(profile_r, [25, 75])],
    }
    return _profile_scores(a, b), pd.DataFrame(rows), summary


# --- positive control: Tozzi-strength biotypes in FOR2107-like data -----------


def _simulate(
    means: np.ndarray,
    proportions: np.ndarray,
    covariance: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    truth = rng.choice(len(means), size=n, p=proportions)
    noise = rng.multivariate_normal(np.zeros(len(covariance)), covariance, size=n)
    return means[truth] + noise, truth


def _mixture_moments(
    means: np.ndarray, proportions: np.ndarray, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = proportions @ means
    centred = means - mean
    return mean, covariance + (centred.T * proportions) @ centred


def _silhouette_task(task: tuple[np.ndarray, int]) -> float:
    from .clustering import _silhouette

    values, k = task
    return _silhouette(fit_clusters(values, k))


def _replicate_task(task: tuple[np.ndarray, np.ndarray, int]) -> tuple[float, float]:
    from .clustering import _silhouette

    values, truth, k = task
    fit = fit_clusters(values, k)
    return _silhouette(fit), adjusted_rand_score(truth, fit.labels)


def power_scenario(
    means: np.ndarray,
    proportions: np.ndarray,
    covariance: np.ndarray,
    columns: np.ndarray,
    n: int,
    k: int,
    replicates: int,
    null_draws: int,
    rng: np.random.Generator,
    executor: ProcessPoolExecutor | None,
) -> dict[str, float]:
    """Detection rate against the multinormal null and recovery of true labels."""
    sub_means = means[:, columns]
    sub_cov = covariance[np.ix_(columns, columns)]
    mix_mean, mix_cov = _mixture_moments(sub_means, proportions, sub_cov)
    nulls = [(rng.multivariate_normal(mix_mean, mix_cov, size=n), k) for _ in range(null_draws)]
    replicates_data = [
        (*_simulate(sub_means, proportions, sub_cov, n, rng), k) for _ in range(replicates)
    ]
    mapper = executor.map if executor else map
    null_scores = np.array(list(mapper(_silhouette_task, nulls, chunksize=8)))
    results = np.array(list(mapper(_replicate_task, replicates_data, chunksize=8)))
    p_values = np.array([(np.sum(null_scores >= s) + 1) / (null_draws + 1) for s in results[:, 0]])
    return {
        "n": n,
        "n_features": int(len(columns)),
        "k": k,
        "median_silhouette": float(np.median(results[:, 0])),
        "null_silhouette_95pct": float(np.percentile(null_scores, 95)),
        "median_p_multinormal": float(np.median(p_values)),
        "power_p_lt_0_05": float(np.mean(p_values < 0.05)),
        "median_ari_true_labels": float(np.median(results[:, 1])),
    }


def run_power_analysis(
    settings: dict[str, Any],
    correlation: pd.DataFrame,
    scenarios: dict[str, dict[str, Any]],
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate planted Tozzi biotypes to his evidence strength, then test FOR2107.

    The planted structure uses Tozzi's circuit-level biotype means (Fig. 3)
    times a scale factor. Because cluster means are inflated by selection,
    the scale is calibrated so that Tozzi's design (41 features, n = 801)
    reproduces his multinormal-null p-value; the as-published scale (1.0) is
    reported as an optimistic bound.
    """
    template = settings["biotype_means"]
    sizes = settings["biotype_sizes"]
    proportions = np.array([sizes[name] for name in template], dtype=float)
    proportions /= proportions.sum()
    circuit_sizes = settings["circuit_feature_counts"]
    labels = [circuit for circuit in CIRCUIT_ORDER for _ in range(circuit_sizes[circuit])]
    feature_circuits = np.asarray(labels)
    # every feature of a circuit gets that circuit's biotype mean; within-
    # biotype noise has unit (control-like) variance and FOR2107 correlations
    means = np.array([[template[name][circuit] for circuit in labels] for name in template])
    covariance = _full_correlation(correlation, labels)
    rng = np.random.default_rng(seed)
    executor = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    calibration_rows = []
    try:
        tozzi = settings["tozzi_design"]
        all_columns = np.arange(len(labels))
        for scale in settings.get("calibration_scales", [0.4, 0.6, 0.8, 1.0]):
            report(f"Power calibration: scale {scale}")
            result = power_scenario(
                means * scale,
                proportions,
                covariance,
                all_columns,
                int(tozzi["n"]),
                int(tozzi["k"]),
                int(settings.get("calibration_replicates", 60)),
                int(settings.get("null_draws", 500)),
                rng,
                executor,
            )
            calibration_rows.append({"scale": scale, **result})
        calibration = pd.DataFrame(calibration_rows)
        target = float(tozzi["p_multinormal"])
        # median p falls with the scale: interpolate scale on increasing log p
        log_p = np.log(calibration.median_p_multinormal.clip(lower=1e-4).to_numpy())
        order = np.argsort(log_p)
        if not log_p.min() <= np.log(target) <= log_p.max():
            report("Power calibration target lies outside the scale grid; using the nearest end")
        calibrated = float(
            np.interp(np.log(target), log_p[order], calibration.scale.to_numpy()[order])
        )
        rows = []
        for scale_name, scale in (("calibrated_to_tozzi_p", calibrated), ("as_published", 1.0)):
            for name, scenario in {"tozzi_design": tozzi, **scenarios}.items():
                circuits = scenario.get("circuits", CIRCUIT_ORDER)
                columns = np.flatnonzero(np.isin(feature_circuits, circuits))
                report(f"Power scenario {name} ({scale_name}, k={scenario['k']})")
                result = power_scenario(
                    means * scale,
                    proportions,
                    covariance,
                    columns,
                    int(scenario["n"]),
                    int(scenario["k"]),
                    int(settings.get("replicates", 200)),
                    int(settings.get("null_draws", 500)),
                    rng,
                    executor,
                )
                rows.append({"scenario": name, "scale": scale_name, "scale_value": scale, **result})
    finally:
        if executor is not None:
            executor.shutdown()
    return calibration, pd.DataFrame(rows)


def _full_correlation(observed: pd.DataFrame, labels: list[str]) -> np.ndarray:
    """41-feature correlation: FOR2107 control correlations where available.

    Circuits without FOR2107 data get the mean within-circuit correlation of
    the available task circuits and no correlation with other features.
    """
    values = observed.to_numpy()
    pools: dict[str, list[int]] = {}
    for index, name in enumerate(observed.columns):
        pools.setdefault(circuit_of(name), []).append(index)
    task_within = [
        values[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
        for circuit, idx in pools.items()
        if circuit not in ("D_C", "S_C", "A_C") and len(idx) > 1
    ]
    fill = float(np.mean(task_within)) if task_within else 0.0
    positions: list[int | None] = []
    used: dict[str, int] = {}
    for circuit in labels:
        index_list = pools.get(circuit, [])
        position = used.get(circuit, 0)
        positions.append(index_list[position] if position < len(index_list) else None)
        used[circuit] = position + 1
    matrix = np.eye(len(labels))
    for i, pi in enumerate(positions):
        for j, pj in enumerate(positions):
            if i == j:
                continue
            if pi is not None and pj is not None:
                matrix[i, j] = values[pi, pj]
            elif labels[i] == labels[j]:
                matrix[i, j] = fill
    # nearest positive-definite correlation by eigenvalue clipping
    eigenvalues, vectors = np.linalg.eigh(matrix)
    matrix = vectors @ np.diag(np.clip(eigenvalues, 1e-3, None)) @ vectors.T
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


# --- orchestration --------------------------------------------------------------


def _prepare_analysis(
    name: str,
    spec: dict[str, Any],
    cohort: pd.DataFrame,
    motion: pd.DataFrame,
    settings: dict[str, Any],
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    table = pd.read_csv(spec["feature_table"], sep="\t")
    groups = select_features(list(table.columns), spec["groups"])
    features = [feature for group in groups.values() for feature in group]
    data = cohort.merge(table[["subject_id", *features]], on="subject_id")
    motion_settings = settings.get("motion", {})
    data, motion_audit = mask_high_motion(
        data,
        groups,
        spec.get("motion_runs", {}) or {},
        motion,
        float(motion_settings.get("max_fraction", 0.25)),
    )
    missing_any = data[features].isna().any(axis=1).to_numpy()
    usable = data[features].notna().any(axis=1).to_numpy()
    data, missing_any = data.loc[usable].reset_index(drop=True), missing_any[usable]
    healthy = data["analysis_role"] == "healthy_reference"
    scores, final, imputation = faithful_scores(
        data,
        features,
        str(settings.get("batch", "site")),
        healthy,
        seed=seed,
        n_jobs=n_jobs,
    )
    clinical = (data["analysis_role"] == "clinical").to_numpy()
    report(
        f"{name}: {int(clinical.sum())} patients, {int(healthy.sum())} controls, "
        f"{len(final)} features; {int(missing_any.sum())} participants had imputed values"
    )
    return {
        "data": data,
        "scores": scores,
        "features": final,
        "clinical": clinical,
        "imputed_any": missing_any,
        "motion_audit": motion_audit,
        "imputation": imputation,
    }


def _clinical_outputs(
    out: Path,
    name: str,
    scores: pd.DataFrame,
    values: pd.DataFrame,
    measures: list[str],
    ks: list[int],
    seed: int,
) -> dict[str, Any]:
    """Cluster profiles, clinical generalization (both tests) and test calibration."""
    matrix = scores.to_numpy(dtype=float)
    features = list(scores.columns)
    summary: dict[str, Any] = {}
    for k in ks:
        labels = fit_clusters(matrix, k).labels
        profiles = cluster_profiles(scores, labels, features)
        profiles.to_csv(out / f"cluster_profiles_k{k}.tsv", sep="\t", index=False)
        summary[f"profiles_k{k}"] = profiles.to_dict(orient="records")
        with stage(f"Clinical generalization for {name}, k={k}"):
            for method in CLINICAL_TESTS:
                table = clinical_generalization(matrix, labels, values, measures, k, seed, method)
                suffix = "" if method == "mann_whitney" else f"_{method}"
                table.to_csv(out / f"clinical_k{k}{suffix}.tsv", sep="\t", index=False)
                summary[f"generalizable_k{k}{suffix}"] = table.loc[
                    table.generalizable_split_half | table.generalizable_leave_site_out,
                    ["cluster", "measure", "r_whole"],
                ].to_dict(orient="records")
            calibration = clinical_test_calibration(values, labels, measures, seed)
            calibration.to_csv(out / f"clinical_test_calibration_k{k}.tsv", sep="\t", index=False)
    return summary


def recompute_clinical(config: Config) -> dict[str, Any]:
    """Redo only the clinical comparisons from saved circuit scores.

    Clustering is deterministic given the scores, so this reproduces the
    full run's cluster labels without repeating the structure tests.
    """
    root = config.require_path("outputs.root")
    settings = config.get("replication", {}) or {}
    seed = int(config.get("project.random_seed", 2107))
    measures = settings.get("clinical_measures", {}) or {}
    cohort, _ = read_normative_cohort(config, measures)
    status: dict[str, Any] = {}
    for name, spec in (settings.get("analyses", {}) or {}).items():
        if not spec.get("evaluate", True):
            continue
        out = root / name
        saved = pd.read_csv(out / "circuit_scores.tsv", sep="\t")
        clinical = saved.loc[saved.analysis_role == "clinical"].reset_index(drop=True)
        features = [column for column in saved.columns[4:]]
        values = clinical[["subject_id"]].merge(
            cohort[["subject_id", "site", *measures]], on="subject_id", how="left"
        )
        candidates = [int(k) for k in spec.get("candidate_clusters", [2, 6])]
        ks = [int(k) for k in settings.get("clinical_k", [2, 6]) if int(k) in candidates]
        status[name] = _clinical_outputs(
            out, name, clinical[features], values, list(measures), ks, seed
        )
    (root / "clinical_status.json").write_text(
        json.dumps(status, indent=2, default=str), encoding="utf-8"
    )
    return status


def run_replication(config: Config) -> dict[str, Any]:
    root = config.require_path("outputs.root")
    root.mkdir(parents=True, exist_ok=True)
    settings = config.get("replication", {}) or {}
    seed = int(config.get("project.random_seed", 2107))
    n_jobs = int(config.get("analysis.n_jobs", 1))
    measures = settings.get("clinical_measures", {}) or {}
    with stage("Load cohort and motion"):
        cohort, exclusions = read_normative_cohort(config, measures)
        clinical_groups = set(settings.get("clinical_groups", []))
        cohort = cohort.loc[
            (cohort.analysis_role == "healthy_reference")
            | cohort.diagnosis_group.isin(clinical_groups)
        ].reset_index(drop=True)
        motion_settings = settings.get("motion", {})
        motion = task_motion_table(
            config.require_path("inputs.task_data_root"),
            cohort.subject_id,
            str(config.get("selection.session", "ses-01")),
            motion_settings.get("runs", ["AP", "FM"]),
            float(motion_settings.get("fd_threshold_mm", 0.3)),
        )
    status: dict[str, Any] = {"mode": "tozzi-faithful-replication", "analyses": {}}
    prepared: dict[str, dict[str, Any]] = {}
    for name, spec in (settings.get("analyses", {}) or {}).items():
        with stage(f"Prepare {name}"):
            prepared[name] = _prepare_analysis(name, spec, cohort, motion, settings, seed, n_jobs)
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        item = prepared[name]
        data, features, clinical = item["data"], item["features"], item["clinical"]
        pd.concat(
            [
                data[["subject_id", "analysis_role", "diagnosis_group", "site"]],
                item["scores"][features],
            ],
            axis=1,
        ).to_csv(out / "circuit_scores.tsv", sep="\t", index=False)
        item["motion_audit"].to_csv(out / "motion_exclusions.tsv", sep="\t", index=False)
        summary: dict[str, Any] = {
            "clinical_n": int(clinical.sum()),
            "healthy_n": int((~clinical).sum()),
            "features": features,
            "participants_with_imputed_values": int(item["imputed_any"].sum()),
        }
        if spec.get("evaluate", True):
            matrix = item["scores"].loc[clinical, features].to_numpy(dtype=float)
            candidates = [int(k) for k in spec.get("candidate_clusters", [2, 6])]
            with stage(f"Structure tests for {name}"):
                structure = evaluate_structure(
                    matrix, candidates, settings.get("iterations", {}), seed, n_jobs
                )
            structure.to_csv(out / "structure.tsv", sep="\t", index=False)
            reference = (settings.get("tozzi_reference", {}) or {}).get(spec.get("reference"), {})
            comparison = benchmark(structure, reference)
            comparison.to_csv(out / "tozzi_benchmark.tsv", sep="\t", index=False)
            summary["structure"] = structure.to_dict(orient="records")
            summary["tozzi_benchmark"] = comparison.to_dict(orient="records")
            ks = [int(k) for k in settings.get("clinical_k", [2, 6]) if int(k) in candidates]
            summary.update(
                _clinical_outputs(
                    out,
                    name,
                    item["scores"].loc[clinical, features].reset_index(drop=True),
                    data.loc[clinical].reset_index(drop=True),
                    list(measures),
                    ks,
                    seed,
                )
            )
        status["analyses"][name] = summary

    reliability = settings.get("reliability", {}) or {}
    if reliability:
        out = root / "reliability"
        out.mkdir(parents=True, exist_ok=True)
        status["reliability"] = {}
        for pair_name, pair in (reliability.get("pairs", {}) or {}).items():
            with stage(f"Measurement agreement: {pair_name}"):
                first, second = prepared[pair["a"]], prepared[pair["b"]]
                frames = []
                for item, suffix in (
                    (first, pair.get("strip_a", "")),
                    (second, pair.get("strip_b", "")),
                ):
                    keep = item["clinical"] & ~item["imputed_any"]
                    frame = item["scores"].loc[keep, item["features"]].copy()
                    frame.index = item["data"].loc[keep, "subject_id"].to_numpy()
                    frame.columns = [column.removesuffix(suffix) for column in frame.columns]
                    frames.append(frame)
                common = frames[0].index.intersection(frames[1].index)
                columns = [c for c in frames[0].columns if c in frames[1].columns]
                per_feature, assignment, summary = measurement_agreement(
                    frames[0].loc[common, columns],
                    frames[1].loc[common, columns],
                    [int(k) for k in reliability.get("candidate_clusters", [2, 6])],
                    seed,
                )
            per_feature.to_csv(out / f"{pair_name}_features.tsv", sep="\t", index=False)
            assignment.to_csv(out / f"{pair_name}_assignment.tsv", sep="\t", index=False)
            status["reliability"][pair_name] = {
                **summary,
                "median_feature_pearson_r": float(per_feature.pearson_r.median()),
                "median_feature_icc_a1": float(per_feature.icc_a1.median()),
                "assignment": assignment.to_dict(orient="records"),
            }

    power = settings.get("power", {}) or {}
    if power:
        source = prepared[power["correlation_from"]]
        healthy = ~source["clinical"]
        correlation = source["scores"].loc[healthy, source["features"]].corr()
        scenarios = {}
        for name, scenario in (power.get("scenarios", {}) or {}).items():
            n = scenario.get("n") or int(prepared[scenario["n_from"]]["clinical"].sum())
            scenarios[name] = {**scenario, "n": n}
        with stage("Positive-control power simulation"):
            calibration, results = run_power_analysis(power, correlation, scenarios, seed, n_jobs)
        out = root / "power"
        out.mkdir(parents=True, exist_ok=True)
        calibration.to_csv(out / "calibration.tsv", sep="\t", index=False)
        results.to_csv(out / "scenarios.tsv", sep="\t", index=False)
        status["power"] = results.to_dict(orient="records")

    exclusions.to_csv(root / "exclusions.tsv", sep="\t", index=False)
    clinical_inputs = [Path(value) for value in config.get("inputs.clinical_files", [])]
    tables = [Path(spec["feature_table"]) for spec in (settings.get("analyses") or {}).values()]
    write_provenance(root / "provenance.json", config.path, [*clinical_inputs, *tables])
    (root / "status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    return status
