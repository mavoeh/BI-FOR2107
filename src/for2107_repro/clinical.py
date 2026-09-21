from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import Config, ConfigError


@dataclass(frozen=True)
class CohortTables:
    all_participants: pd.DataFrame
    clinical: pd.DataFrame
    healthy: pd.DataFrame
    exclusions: pd.DataFrame


def load_variable_map(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        mapping = yaml.safe_load(stream)
    if not isinstance(mapping, dict):
        raise ConfigError("Variable map must be a mapping")
    return mapping


def _required_columns(mapping: dict[str, Any]) -> list[str]:
    scalar = [mapping.get(key) for key in ("subject_id", "session", "diagnosis", "site", "scanner", "age", "sex")]
    measures = list((mapping.get("clinical_measures") or {}).values())
    indicators = list((mapping.get("healthy_exclusion_indicators") or {}).keys())
    return sorted({column for column in scalar + measures + indicators if column})


def read_clinical(config: Config) -> pd.DataFrame:
    variable_map_path = Path(config.get("selection.variable_map"))
    mapping = load_variable_map(variable_map_path)
    columns = _required_columns(mapping)
    if not columns or not mapping.get("subject_id") or not mapping.get("diagnosis"):
        raise ConfigError(
            f"Complete subject_id and diagnosis in {variable_map_path}; SPSS inventory lists candidates."
        )
    try:
        import pyreadstat
    except ImportError as exc:
        raise ConfigError("pyreadstat is required to read the clinical database") from exc
    source = config.get("inputs.primary_clinical_file")
    if not source:
        raise ConfigError("inputs.primary_clinical_file is required for the baseline analysis")
    data, _ = pyreadstat.read_sav(source, usecols=columns, apply_value_formats=True)
    data["source_file"] = Path(source).name
    for semantic in ("subject_id", "session", "diagnosis", "site", "scanner", "age", "sex"):
        source_column = mapping.get(semantic)
        if source_column:
            data[semantic] = data[source_column]
    for semantic, source_column in (mapping.get("clinical_measures") or {}).items():
        data[semantic] = data[source_column]
    prefix = config.get("selection.subject_id_prefix", "sub-")
    width = int(config.get("selection.subject_id_width", 4))
    normalized = data["subject_id"].astype(str).str.strip().str.removeprefix(prefix).str.zfill(width)
    data["subject_id"] = prefix + normalized
    if "session" not in data:
        data["session"] = config.get("selection.session", "ses-01")
    return data


def select_cohorts(data: pd.DataFrame, config: Config) -> CohortTables:
    variable_map = load_variable_map(Path(config.get("selection.variable_map")))
    diagnosis_groups = variable_map.get("diagnosis_groups") or {}
    requested = config.get("selection.clinical_diagnoses", [])
    clinical_values = {value for group in requested for value in diagnosis_groups.get(group, [])}
    healthy_values = set(diagnosis_groups.get("HC", [])) | set(config.get("selection.healthy_labels", []))
    session = config.get("selection.session")
    included = data.copy()
    exclusions: list[dict[str, str]] = []
    if "session" in included and session:
        wrong = included["session"].astype(str) != session
        exclusions.extend(
            {"subject_id": str(value), "reason": "non-primary-session"}
            for value in included.loc[wrong, "subject_id"]
        )
        included = included.loc[~wrong]
    diagnoses = included["diagnosis"].astype(str)
    healthy_mask = diagnoses.isin({str(value) for value in healthy_values})
    for column, positive_values in (variable_map.get("healthy_exclusion_indicators") or {}).items():
        if column in included:
            healthy_mask &= ~included[column].astype(str).isin({str(value) for value in positive_values})
    if config.get("selection.include_all_diagnoses", False):
        clinical_mask = ~healthy_mask
    else:
        clinical_mask = diagnoses.isin({str(value) for value in clinical_values})
    excluded_mask = ~(clinical_mask | healthy_mask)
    exclusions.extend(
        {"subject_id": str(value), "reason": "diagnosis-not-selected"}
        for value in included.loc[excluded_mask, "subject_id"]
    )
    return CohortTables(
        all_participants=included.loc[~excluded_mask].copy(),
        clinical=included.loc[clinical_mask].copy(),
        healthy=included.loc[healthy_mask].copy(),
        exclusions=pd.DataFrame(exclusions, columns=["subject_id", "reason"]),
    )


def write_cohort_tables(
    config: Config,
    imaging_subjects: set[str] | None = None,
    imaging_records: pd.DataFrame | None = None,
) -> dict[str, Any]:
    data = read_clinical(config)
    cohorts = select_cohorts(data, config)
    output = config.require_path("outputs.root") / "cohort"
    output.mkdir(parents=True, exist_ok=True)
    selected = cohorts.all_participants.copy()
    if imaging_records is not None and not imaging_records.empty and "scanner" in imaging_records:
        scanner_map = (
            imaging_records.dropna(subset=["scanner"])
            .groupby("subject")["scanner"]
            .agg(lambda values: "|".join(sorted({str(value) for value in values})))
        )
        selected["scanner"] = selected["subject_id"].map(scanner_map).fillna(selected.get("scanner"))
    selected["imaging_available"] = True
    exclusions = cohorts.exclusions.copy()
    if imaging_subjects is not None:
        selected["imaging_available"] = selected["subject_id"].isin(imaging_subjects)
        unavailable = selected.loc[~selected["imaging_available"], "subject_id"]
        extra = pd.DataFrame({"subject_id": unavailable, "reason": "no-configured-session-imaging"})
        exclusions = pd.concat([exclusions, extra], ignore_index=True)
        selected = selected.loc[selected["imaging_available"]].copy()
    clinical_ids = set(cohorts.clinical["subject_id"])
    healthy_ids = set(cohorts.healthy["subject_id"])
    selected["analysis_role"] = selected["subject_id"].map(
        lambda value: "clinical" if value in clinical_ids else "healthy_reference" if value in healthy_ids else "excluded"
    )
    safe_columns = [
        column
        for column in (
            "subject_id",
            "session",
            "analysis_role",
            "diagnosis",
            "site",
            "scanner",
            "age",
            "sex",
            "bdi_total",
            "hama_total",
            "hamd17_total",
            "hamd21_total",
            "imaging_available",
        )
        if column in selected
    ]
    selected[safe_columns].to_csv(output / "participants.tsv", sep="\t", index=False)
    exclusions.to_csv(output / "exclusions.tsv", sep="\t", index=False)
    summary = {
        "selected_with_imaging": len(selected),
        "clinical_with_imaging": int((selected["analysis_role"] == "clinical").sum()),
        "healthy_reference_with_imaging": int((selected["analysis_role"] == "healthy_reference").sum()),
        "excluded_records": len(exclusions),
    }
    return {"summary": summary, "participants": selected[safe_columns], "exclusions": exclusions}


def characterize_clusters(
    data: pd.DataFrame, cluster_column: str, measures: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster in sorted(data[cluster_column].dropna().unique()):
        inside = data[cluster_column] == cluster
        for measure in measures:
            left = data.loc[inside, measure].dropna()
            right = data.loc[~inside, measure].dropna()
            if len(left) < 2 or len(right) < 2:
                continue
            test = stats.mannwhitneyu(
                left,
                right,
                alternative="two-sided",
                method="asymptotic",
            )
            statistic, p_value = test.statistic, test.pvalue
            rank_biserial = 2 * statistic / (len(left) * len(right)) - 1
            rows.append(
                {
                    "cluster": int(cluster),
                    "measure": measure,
                    "n_inside": len(left),
                    "n_outside": len(right),
                    "median_inside": left.median(),
                    "median_outside": right.median(),
                    "u": statistic,
                    "effect_r": rank_biserial,
                    "p_value": p_value,
                }
            )
    results = pd.DataFrame(rows)
    if not results.empty:
        results["p_fdr"] = multipletests(results["p_value"], method="fdr_bh")[1]
    return results
