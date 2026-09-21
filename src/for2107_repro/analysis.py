from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clinical import characterize_clusters
from .clustering import (
    evaluate_solutions,
    fit_clusters,
    leave_one_out_stability,
    null_silhouette_tests,
    repeated_split_half_replication,
    split_half_replication,
    subsample_stability,
)
from .config import Config
from .harmonize import combat_harmonize, healthy_reference_scores, random_forest_pmm
from .progress import report, stage
from .validation import (
    evaluate_selection_criteria,
    leave_site_out_replication,
    nuisance_associations,
)

NON_FEATURE_COLUMNS = {
    "subject_id",
    "qc_pass",
    "high_motion_fraction",
    "motion_qc_available",
    "analysis_role",
    "diagnosis",
    "site",
    "scanner",
    "age",
    "sex",
    "session",
    "imaging_available",
    "bdi_total",
    "hama_total",
    "hamd17_total",
    "hamd21_total",
}


def _encode_covariates(data: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    result = data.copy()
    encoded = []
    for name in names:
        if name not in result:
            continue
        if pd.api.types.is_numeric_dtype(result[name]):
            encoded.append(name)
        else:
            codes, _ = pd.factorize(result[name], sort=True)
            encoded_name = f"{name}_code"
            result[encoded_name] = codes
            encoded.append(encoded_name)
    return result, encoded


def run_primary_analysis(
    config: Config, features: pd.DataFrame, participants: pd.DataFrame
) -> dict[str, Any]:
    output = config.require_path("outputs.root") / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    if features.empty:
        return _write_gate(output, ["no-extracted-features"])
    data = participants.merge(features, on="subject_id", how="inner")
    feature_names = [column for column in features if column not in NON_FEATURE_COLUMNS]
    feature_names = [column for column in feature_names if data[column].notna().any()]
    reasons = []
    if len(feature_names) < 2:
        reasons.append("fewer-than-two-usable-features")
    if "qc_pass" in data:
        include_unassessed = (
            config.get("preprocessing.missing_motion_policy", "exclude") == "include_unassessed"
        )
        data = data.loc[data["qc_pass"].fillna(include_unassessed)]
    clinical_n = int((data["analysis_role"] == "clinical").sum())
    healthy_n = int((data["analysis_role"] == "healthy_reference").sum())
    if clinical_n < int(config.get("analysis.minimum_clustering_n", 30)):
        reasons.append(f"clinical-n-{clinical_n}-below-minimum")
    if healthy_n < int(config.get("analysis.minimum_reference_n", 20)):
        reasons.append(f"healthy-n-{healthy_n}-below-minimum")
    if reasons:
        report(f"Analysis gated: {', '.join(reasons)}")
        return _write_gate(output, reasons, clinical_n=clinical_n, healthy_n=healthy_n)
    batch_column = "scanner" if "scanner" in data and data["scanner"].notna().any() else "site"
    report(
        f"Analysis input: {clinical_n} clinical, {healthy_n} reference, "
        f"{len(feature_names)} features"
    )
    n_jobs = int(config.get("analysis.n_jobs", 1))
    with stage("Impute missing feature values"):
        imputed, imputation_audit = random_forest_pmm(
            data,
            feature_names,
            batch_column,
            seed=int(config.get("project.random_seed", 2107)),
            iterations=int(config.get("analysis.imputation_iterations", 1)),
            n_jobs=n_jobs,
        )
    with stage("Harmonize scanner/site effects"):
        encoded, covariates = _encode_covariates(imputed, ["age", "sex"])
        harmonized = combat_harmonize(encoded, feature_names, batch_column, covariates)
    with stage("Reference circuit scores to healthy controls"):
        normalized, reference_audit = healthy_reference_scores(
            harmonized,
            feature_names,
            harmonized["analysis_role"] == "healthy_reference",
            minimum_n=int(config.get("analysis.minimum_reference_n", 20)),
        )
    clinical = normalized.loc[normalized["analysis_role"] == "clinical"].copy()
    matrix = clinical[feature_names]
    candidates = config.get("analysis.candidate_clusters", list(range(2, 16)))
    candidates = [value for value in candidates if value < len(clinical)]
    report(f"Statistical validation workers: {'all available' if n_jobs == -1 else n_jobs}")
    with stage("Evaluate candidate cluster solutions"):
        evaluation = evaluate_solutions(matrix, candidates)
    with stage("Run silhouette null tests"):
        nulls = null_silhouette_tests(
            matrix,
            candidates,
            simulations=int(config.get("analysis.null_iterations", 10000)),
            permutations=int(config.get("analysis.permutation_iterations", 10000)),
            seed=int(config.get("project.random_seed", 2107)),
            n_jobs=n_jobs,
        )
    with stage("Run leave-one-out stability"):
        loo = leave_one_out_stability(matrix, candidates, n_jobs=n_jobs)
    with stage("Run participant-subsample stability"):
        subsample = subsample_stability(
            matrix,
            candidates,
            fraction=float(config.get("analysis.subsample_fraction", 0.8)),
            iterations=int(config.get("analysis.subsample_iterations", 10000)),
            seed=int(config.get("project.random_seed", 2107)),
            n_jobs=n_jobs,
        )
    combined_evaluation = evaluation.merge(nulls, on=["n_clusters", "silhouette"], how="left")
    with stage("Apply prespecified cluster-solution criteria"):
        selection_table, selection_status = evaluate_selection_criteria(
            combined_evaluation,
            loo,
            subsample,
            config.get(
                "analysis.selection_thresholds",
                {
                    "p_multinormal": 0.05,
                    "p_permuted": 0.05,
                    "loo_ari_mean": 0.75,
                    "subsample_ari_mean": 0.28,
                },
            ),
            confirmatory_clusters=int(
                config.get(
                    "analysis.confirmatory_clusters",
                    config.get("analysis.selected_clusters", 6),
                )
            ),
        )
    selected = int(config.get("analysis.selected_clusters", 6))
    with stage(f"Fit selected {selected}-cluster solution and split-half check"):
        fit = fit_clusters(matrix, selected)
        clinical["cluster"] = fit.labels
        np.save(output / "linkage_matrix.npy", fit.linkage_matrix)
        np.save(output / "correlation_distance_matrix.npy", fit.distance_matrix)
        split = split_half_replication(
            matrix, selected, seed=int(config.get("project.random_seed", 2107))
        )
    split_iterations = int(config.get("analysis.split_half_iterations", 1))
    with stage(f"Repeat split-half replication {split_iterations} times"):
        repeated_split = repeated_split_half_replication(
            matrix,
            selected,
            iterations=split_iterations,
            seed=int(config.get("project.random_seed", 2107)),
            n_jobs=n_jobs,
        )
        split_summary = (
            repeated_split.groupby("cluster")["profile_correlation"]
            .agg(
                iterations="count",
                mean="mean",
                sd="std",
                median="median",
                minimum="min",
                maximum="max",
            )
            .reset_index()
        )
        quantiles = (
            repeated_split.groupby("cluster")["profile_correlation"]
            .quantile([0.025, 0.975])
            .unstack()
            .rename(columns={0.025: "ci_2_5", 0.975: "ci_97_5"})
            .reset_index()
        )
        split_summary = split_summary.merge(quantiles, on="cluster", how="left")
    leave_site = pd.DataFrame()
    if bool(config.get("analysis.leave_site_out", False)):
        with stage("Run bidirectional leave-site-out replication"):
            leave_site = leave_site_out_replication(
                imputed,
                feature_names,
                batch_column,
                n_clusters=selected,
                minimum_reference_n=int(config.get("analysis.minimum_reference_n", 20)),
            )
    nuisance = pd.DataFrame()
    if bool(config.get("analysis.nuisance_checks", False)):
        with stage("Test nuisance-variable balance between clusters"):
            nuisance = nuisance_associations(clinical, "cluster")
    with stage("Test clinical differences between clusters"):
        measure_names = [
            name
            for name in ("bdi_total", "hama_total", "hamd17_total", "hamd21_total")
            if name in clinical
        ]
        clinical_results = characterize_clusters(clinical, "cluster", measure_names)
    tables = {
        "imputation_audit": imputation_audit,
        "reference_audit": reference_audit,
        "solution_evaluation": combined_evaluation,
        "solution_selection": selection_table,
        "loo_stability": loo,
        "subsample_stability": subsample,
        "split_half_iterations": repeated_split,
        "split_half_summary": split_summary,
        "leave_site_out": leave_site,
        "nuisance_associations": nuisance,
        "cluster_assignments": clinical[["subject_id", "cluster", *feature_names]],
        "clinical_results": clinical_results,
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.tsv", sep="\t", index=False)
    status = {
        "ready": True,
        "clinical_n": clinical_n,
        "healthy_n": healthy_n,
        "features": feature_names,
        "selected_clusters": selected,
        "split_half_mean_profile_correlation": split["mean_profile_correlation"],
        "split_half_iterations": split_iterations,
        "selection": selection_status,
        "leave_site_out_ready": not leave_site.empty,
        "nuisance_checks_ready": not nuisance.empty,
    }
    (output / "solution_selection_status.json").write_text(
        json.dumps(selection_status, indent=2), encoding="utf-8"
    )
    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def _write_gate(output: Path, reasons: list[str], **counts: int) -> dict[str, Any]:
    status: dict[str, Any] = {"ready": False, "reasons": reasons, **counts}
    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status
