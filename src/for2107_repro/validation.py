from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from .clustering import assign_by_profile_correlation, fit_clusters


def evaluate_selection_criteria(
    evaluation: pd.DataFrame,
    loo: pd.DataFrame,
    subsample: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    confirmatory_clusters: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply explicit statistical and stability gates to every candidate k."""
    table = evaluation.merge(loo, on="n_clusters", how="left").merge(
        subsample, on="n_clusters", how="left"
    )
    criteria = {
        "passes_multinormal": ("p_multinormal", "lt"),
        "passes_permutation": ("p_permuted", "lt"),
        "passes_loo": ("loo_ari_mean", "ge"),
        "passes_subsample": ("subsample_ari_mean", "ge"),
    }
    for output, (column, operation) in criteria.items():
        threshold = float(thresholds[column])
        table[output] = (
            table[column] < threshold if operation == "lt" else table[column] >= threshold
        )
    pass_columns = list(criteria)
    table["passes_all_prespecified_criteria"] = table[pass_columns].all(axis=1)
    table["confirmatory_solution"] = table["n_clusters"] == confirmatory_clusters
    supported = table.loc[table["passes_all_prespecified_criteria"]].sort_values(
        ["silhouette", "n_clusters"], ascending=[False, True]
    )
    selected = int(supported.iloc[0]["n_clusters"]) if not supported.empty else None
    confirmatory = table.loc[table["confirmatory_solution"]]
    status = {
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "confirmatory_clusters": int(confirmatory_clusters),
        "confirmatory_passes_all": bool(
            not confirmatory.empty and confirmatory.iloc[0]["passes_all_prespecified_criteria"]
        ),
        "supported_candidates": supported["n_clusters"].astype(int).tolist(),
        "highest_silhouette_supported_candidate": selected,
        "selection_is_automatic": False,
        "selection_note": (
            "Candidates must additionally be reviewed for elbow, profile replication, "
            "site generalization, nuisance balance, and theoretical interpretability."
        ),
    }
    return table, status


def nuisance_associations(
    data: pd.DataFrame,
    cluster_column: str,
    *,
    categorical: Iterable[str] = ("site", "scanner", "sex"),
    continuous: Iterable[str] = ("age",),
) -> pd.DataFrame:
    """Test whether cluster membership is associated with nuisance variables."""
    rows: list[dict[str, Any]] = []
    for name in categorical:
        if name not in data or data[name].dropna().nunique() < 2:
            continue
        table = pd.crosstab(data[cluster_column], data[name])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        statistic, p_value, _, _ = stats.chi2_contingency(table)
        n = int(table.to_numpy().sum())
        denominator = max(1, min(table.shape) - 1)
        cramers_v = float(np.sqrt(statistic / (n * denominator)))
        rows.append(
            {
                "variable": name,
                "test": "chi_square",
                "n": n,
                "statistic": float(statistic),
                "effect_size": cramers_v,
                "effect_name": "cramers_v",
                "p_value": float(p_value),
            }
        )
    for name in continuous:
        if name not in data:
            continue
        groups = [
            pd.to_numeric(group[name], errors="coerce").dropna().to_numpy()
            for _, group in data.groupby(cluster_column)
        ]
        groups = [group for group in groups if len(group) >= 2]
        if len(groups) < 2:
            continue
        statistic, p_value = stats.f_oneway(*groups)
        values = np.concatenate(groups)
        grand_mean = float(values.mean())
        between = sum(len(group) * (float(group.mean()) - grand_mean) ** 2 for group in groups)
        total = float(np.sum((values - grand_mean) ** 2))
        eta_squared = float(between / total) if total > 0 else 0.0
        rows.append(
            {
                "variable": name,
                "test": "one_way_anova",
                "n": len(values),
                "statistic": float(statistic),
                "effect_size": eta_squared,
                "effect_name": "eta_squared",
                "p_value": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_fdr"] = stats.false_discovery_control(result["p_value"], method="bh")
    return result


def _reference_standardize(
    data: pd.DataFrame,
    feature_names: list[str],
    reference_mask: pd.Series,
    minimum_reference_n: int,
) -> pd.DataFrame:
    reference = data.loc[reference_mask, feature_names]
    if len(reference) < minimum_reference_n:
        raise ValueError(f"Site reference n={len(reference)} below minimum {minimum_reference_n}")
    means = reference.mean()
    standard_deviations = reference.std(ddof=1)
    if standard_deviations.isna().any() or (standard_deviations <= 0).any():
        raise ValueError("Site reference contains an invalid feature standard deviation")
    result = data.copy()
    result.loc[:, feature_names] = (data[feature_names] - means) / standard_deviations
    return result


def leave_site_out_replication(
    data: pd.DataFrame,
    feature_names: list[str],
    site_column: str,
    *,
    n_clusters: int,
    minimum_reference_n: int,
) -> pd.DataFrame:
    """Train profiles at one site and test assignments at another site.

    Each site is independently referenced to its own healthy controls. This
    avoids estimating ComBat parameters with a held-out acquisition batch.
    """
    sites = sorted(data[site_column].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    for training_site in sites:
        for test_site in sites:
            if training_site == test_site:
                continue
            training = data.loc[data[site_column].astype(str) == training_site].copy()
            testing = data.loc[data[site_column].astype(str) == test_site].copy()
            training = _reference_standardize(
                training,
                feature_names,
                training["analysis_role"] == "healthy_reference",
                minimum_reference_n,
            )
            testing = _reference_standardize(
                testing,
                feature_names,
                testing["analysis_role"] == "healthy_reference",
                minimum_reference_n,
            )
            train_clinical = training.loc[training["analysis_role"] == "clinical"]
            test_clinical = testing.loc[testing["analysis_role"] == "clinical"]
            train_matrix = train_clinical[feature_names].to_numpy(dtype=float)
            test_matrix = test_clinical[feature_names].to_numpy(dtype=float)
            train_fit = fit_clusters(train_matrix, n_clusters)
            train_profiles = np.vstack(
                [
                    train_matrix[train_fit.labels == cluster].mean(axis=0)
                    for cluster in range(1, n_clusters + 1)
                ]
            )
            transferred_labels = assign_by_profile_correlation(test_matrix, train_profiles)
            independent_labels = fit_clusters(test_matrix, n_clusters).labels
            assignment_ari = float(adjusted_rand_score(independent_labels, transferred_labels))
            for cluster in range(1, n_clusters + 1):
                selected = transferred_labels == cluster
                correlation = (
                    float(
                        np.corrcoef(
                            train_profiles[cluster - 1],
                            test_matrix[selected].mean(axis=0),
                        )[0, 1]
                    )
                    if selected.any()
                    else np.nan
                )
                rows.append(
                    {
                        "training_site": training_site,
                        "test_site": test_site,
                        "cluster": cluster,
                        "training_clinical_n": len(train_clinical),
                        "test_clinical_n": len(test_clinical),
                        "test_assigned_n": int(selected.sum()),
                        "assignment_ari_vs_independent_test_clustering": assignment_ari,
                        "profile_correlation": correlation,
                        "normalization": "within-site healthy reference",
                    }
                )
    return pd.DataFrame(rows)
