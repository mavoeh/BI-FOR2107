"""Normative deviation modelling of circuit features against healthy controls.

This is the dimensional alternative to biotype clustering. Scanner effects are
harmonized with ComBat learned on healthy controls only, using scanner epochs
(site x hardware/sequence period) instead of sites as batches. Each feature is
then modelled in controls as a function of age (cubic B-spline) and sex, with
heteroscedastic noise and a sinh-arcsinh shape correction for heavy tails.
Controls receive out-of-fold deviation scores; patients are scored with the
model fitted on all controls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer
from statsmodels.stats.multitest import multipletests

from .clinical import load_variable_map, read_clinical, select_cohorts
from .config import Config, ConfigError
from .motion import motion_exclusion_reasons, task_motion_table
from .progress import report, stage
from .provenance import write_provenance

NON_FEATURE_COLUMNS = {"subject_id", "n_ap_volumes", "n_fm_volumes"}
COVARIATE_COLUMNS = [
    "subject_id",
    "analysis_role",
    "diagnosis_group",
    "site",
    "scanner_epoch",
    "age",
    "female",
]


def assign_scanner_epochs(codes: pd.DataFrame, rules: dict[str, dict[str, float]]) -> pd.Series:
    """Label each row with the single epoch rule whose column codes all match.

    Rows matching no rule are left missing; a row matching several rules is a
    configuration error because the batches would no longer be disjoint.
    """
    matches = pd.DataFrame(index=codes.index)
    for label, conditions in rules.items():
        missing = [column for column in conditions if column not in codes]
        if missing:
            raise ConfigError(f"Scanner-epoch rule {label} uses unknown columns: {missing}")
        mask = pd.Series(True, index=codes.index)
        for column, value in conditions.items():
            mask &= codes[column] == float(value)
        matches[label] = mask
    overlapping = matches.sum(axis=1) > 1
    if overlapping.any():
        raise ConfigError(f"{int(overlapping.sum())} rows match more than one scanner-epoch rule")
    labels = matches.idxmax(axis=1).where(matches.any(axis=1))
    return labels.astype(object)


def _read_numeric_columns(config: Config, columns: list[str]) -> pd.DataFrame:
    import pyreadstat

    source = config.get("inputs.primary_clinical_file")
    mapping = load_variable_map(Path(config.get("selection.variable_map")))
    subject_column = mapping["subject_id"]
    data, _ = pyreadstat.read_sav(
        source, usecols=sorted({subject_column, *columns}), apply_value_formats=False
    )
    prefix = config.get("selection.subject_id_prefix", "sub-")
    width = int(config.get("selection.subject_id_width", 4))
    identifiers = data[subject_column].astype(str).str.replace(r"\.0$", "", regex=True)
    data["subject_id"] = prefix + identifiers.str.strip().str.removeprefix(prefix).str.zfill(width)
    return data.drop(columns=[subject_column]) if subject_column not in columns else data


def read_normative_cohort(
    config: Config, measures: dict[str, str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble roles, diagnosis groups, covariates, scanner epochs and measures.

    `measures` maps output names to SPSS columns; it defaults to
    `normative.clinical_measures`.
    """
    clinical = read_clinical(config)
    cohort = select_cohorts(clinical, config)
    healthy_ids = set(cohort.healthy.subject_id)
    labels = config.get("selection.diagnosis_groups", {})
    if not labels:
        raise ConfigError("selection.diagnosis_groups must map database labels to groups")
    epoch_rules = config.get("harmonization.scanner_epochs", {})
    if not epoch_rules:
        raise ConfigError("harmonization.scanner_epochs must define at least one epoch")
    flags = config.get("selection.exclusion_flags", {}) or {}
    if measures is None:
        measures = config.get("normative.clinical_measures", {}) or {}
    mapping = load_variable_map(Path(config.get("selection.variable_map")))
    epoch_columns = sorted({column for rule in epoch_rules.values() for column in rule})
    numeric = _read_numeric_columns(
        config, [*epoch_columns, *flags, *measures.values(), mapping["sex"]]
    )

    data = clinical[["subject_id", "diagnosis", "site", "age"]].merge(
        numeric, on="subject_id", how="left"
    )
    data["diagnosis_group"] = data["diagnosis"].astype(str).map(labels)
    data["female"] = (data[mapping["sex"]] == float(config.get("normative.female_code", 2))).astype(
        float
    )
    data.loc[data[mapping["sex"]].isna(), "female"] = np.nan
    data["scanner_epoch"] = assign_scanner_epochs(data[epoch_columns], epoch_rules)
    data["analysis_role"] = np.where(
        data.subject_id.isin(healthy_ids),
        "healthy_reference",
        np.where(data.diagnosis_group.notna() & (data.diagnosis_group != "HC"), "clinical", ""),
    )
    for name, column in measures.items():
        data[name] = data[column]

    reasons = pd.Series("", index=data.index)
    for column, values in flags.items():
        flagged = data[column].isin([float(value) for value in values])
        reasons = reasons.where(~flagged | (reasons != ""), f"exclusion-flag-{column}")
    checks = [
        (data.analysis_role == "", "not-healthy-reference-and-no-clinical-group"),
        (data.age.isna(), "age-missing"),
        (data.female.isna(), "sex-missing"),
        (data.scanner_epoch.isna(), "scanner-epoch-unknown"),
    ]
    for mask, reason in checks:
        reasons = reasons.where(~mask | (reasons != ""), reason)
    exclusions = data.loc[reasons != "", ["subject_id", "diagnosis_group"]].assign(
        reason=reasons[reasons != ""]
    )
    keep = [*COVARIATE_COLUMNS, *measures]
    return data.loc[reasons == "", keep].reset_index(drop=True), exclusions.reset_index(drop=True)


def _combat_covariates(data: pd.DataFrame, age_center: float, protected: list[str]) -> pd.DataFrame:
    centered = data["age"].to_numpy(dtype=float) - age_center
    covars = pd.DataFrame(
        {
            "SITE": data["scanner_epoch"].astype(str).to_numpy(),
            "AGE": centered,
            "AGE2": centered**2,
            "FEMALE": data["female"].to_numpy(dtype=float),
        }
    )
    for name in protected:
        covars[name.upper()] = data[name].to_numpy(dtype=float)
    return covars


@dataclass
class HealthyComBat:
    """ComBat learned on healthy controls and applied unchanged to others.

    Age, age², sex and any `protected` covariates (e.g. motion) are modelled so
    that their differences between epochs are not removed as batch effects.
    """

    age_center: float = 40.0
    protected: list[str] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)

    def fit_transform(self, data: pd.DataFrame, features: list[str]) -> np.ndarray:
        from neuroHarmonize import harmonizationLearn

        covars = _combat_covariates(data, self.age_center, self.protected)
        if covars["SITE"].nunique() <= 1:
            self.model = {}
            return data[features].to_numpy(dtype=float)
        self.model, adjusted = harmonizationLearn(data[features].to_numpy(dtype=float), covars)
        return adjusted

    def transform(self, data: pd.DataFrame, features: list[str]) -> np.ndarray:
        from neuroHarmonize import harmonizationApply

        if not self.model:
            return data[features].to_numpy(dtype=float)
        unseen = set(data["scanner_epoch"].astype(str)) - set(self.model["SITE_labels"])
        if unseen:
            raise ValueError(f"Scanner epochs without healthy reference data: {sorted(unseen)}")
        covars = _combat_covariates(data, self.age_center, self.protected)
        return harmonizationApply(data[features].to_numpy(dtype=float), covars, self.model)


def _shash_transform(values: np.ndarray, params: np.ndarray) -> np.ndarray:
    location, log_scale, skew, log_tail = params
    scaled = (values - location) / np.exp(log_scale)
    return np.sinh(np.exp(log_tail) * np.arcsinh(scaled) - skew)


def _shash_negative_loglik(params: np.ndarray, values: np.ndarray) -> float:
    location, log_scale, skew, log_tail = params
    tail = np.exp(log_tail)
    scaled = (values - location) / np.exp(log_scale)
    inner = tail * np.arcsinh(scaled) - skew
    log_jacobian = np.log(tail) + np.log(np.cosh(inner)) - 0.5 * np.log1p(scaled**2) - log_scale
    return float(-np.sum(stats.norm.logpdf(np.sinh(inner)) + log_jacobian))


def fit_shash(values: np.ndarray) -> np.ndarray:
    """Maximum-likelihood sinh-arcsinh (Jones & Pewsey) fit to residuals.

    Returns (location, log scale, skewness, log tail weight); a tail weight
    below one means heavier-than-Gaussian tails.
    """
    values = np.asarray(values, dtype=float)
    start = np.array([0.0, np.log(values.std(ddof=1)), 0.0, 0.0])
    bounds = [(None, None), (None, None), (-3.0, 3.0), (np.log(0.2), np.log(5.0))]
    result = optimize.minimize(
        _shash_negative_loglik, start, args=(values,), method="L-BFGS-B", bounds=bounds
    )
    if not np.all(np.isfinite(result.x)):
        raise ValueError("Sinh-arcsinh fit did not converge to finite parameters")
    return result.x


@dataclass
class NormativeFeatureModel:
    """Heteroscedastic B-spline age model with sinh-arcsinh shape correction.

    `linear` holds covariates entered linearly in both the mean and the
    log-variance model: sex, optionally followed by e.g. log mean FD.
    """

    age_knots: int = 4
    variance_iterations: int = 3
    spline: SplineTransformer | None = None
    mean_coef: np.ndarray | None = None
    mean_cov: np.ndarray | None = None
    variance_coef: np.ndarray | None = None
    shape: np.ndarray | None = None

    def _design(self, age: np.ndarray, linear: np.ndarray) -> np.ndarray:
        assert self.spline is not None
        basis = self.spline.transform(np.asarray(age, dtype=float)[:, None])
        linear = np.asarray(linear, dtype=float).reshape(len(basis), -1)
        return np.column_stack([np.ones(len(basis)), basis, linear])

    def fit(self, age: np.ndarray, linear: np.ndarray, y: np.ndarray) -> NormativeFeatureModel:
        import statsmodels.api as sm

        y = np.asarray(y, dtype=float)
        self.spline = SplineTransformer(
            n_knots=self.age_knots,
            degree=3,
            knots="quantile",
            extrapolation="linear",
            include_bias=False,
        ).fit(np.asarray(age, dtype=float)[:, None])
        design = self._design(age, linear)
        weights = np.ones(len(y))
        floor = 1e-10 * float(np.var(y))
        for _ in range(self.variance_iterations):
            weighted = design * weights[:, None]
            self.mean_cov = np.linalg.pinv(design.T @ weighted)
            self.mean_coef = self.mean_cov @ (weighted.T @ y)
            squared = np.maximum((y - design @ self.mean_coef) ** 2, floor)
            fit = sm.GLM(
                squared, design, family=sm.families.Gamma(link=sm.families.links.Log())
            ).fit()
            self.variance_coef = np.asarray(fit.params)
            weights = 1.0 / np.exp(design @ self.variance_coef)
        weighted = design * weights[:, None]
        self.mean_cov = np.linalg.pinv(design.T @ weighted)
        self.mean_coef = self.mean_cov @ (weighted.T @ y)
        self.shape = fit_shash(self._standardized(age, linear, y))
        return self

    def predict(self, age: np.ndarray, linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted mean and total predictive standard deviation."""
        design = self._design(age, linear)
        mean = design @ self.mean_coef
        noise = np.exp(design @ self.variance_coef)
        parameter = np.einsum("ij,jk,ik->i", design, self.mean_cov, design)
        return mean, np.sqrt(noise + parameter)

    def _standardized(self, age: np.ndarray, linear: np.ndarray, y: np.ndarray) -> np.ndarray:
        mean, sd = self.predict(age, linear)
        return (np.asarray(y, dtype=float) - mean) / sd

    def zscore(self, age: np.ndarray, linear: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert self.shape is not None
        return _shash_transform(self._standardized(age, linear, y), self.shape)


def _linear_names(settings: dict[str, Any]) -> list[str]:
    return ["female", *(settings.get("linear_covariates") or [])]


def _linear_terms(data: pd.DataFrame, settings: dict[str, Any]) -> np.ndarray:
    return data[_linear_names(settings)].to_numpy(dtype=float)


def _combat(settings: dict[str, Any], age_center: float) -> HealthyComBat:
    return HealthyComBat(
        age_center=age_center, protected=list(settings.get("linear_covariates") or [])
    )


def _fit_models(
    data: pd.DataFrame, harmonized: np.ndarray, features: list[str], settings: dict[str, Any]
) -> dict[str, NormativeFeatureModel]:
    age = data["age"].to_numpy(dtype=float)
    linear = _linear_terms(data, settings)
    return {
        feature: NormativeFeatureModel(
            age_knots=int(settings.get("age_knots", 4)),
            variance_iterations=int(settings.get("variance_iterations", 3)),
        ).fit(age, linear, harmonized[:, index])
        for index, feature in enumerate(features)
    }


def _score(
    data: pd.DataFrame,
    harmonized: np.ndarray,
    models: dict[str, NormativeFeatureModel],
    features: list[str],
    settings: dict[str, Any],
) -> pd.DataFrame:
    age = data["age"].to_numpy(dtype=float)
    linear = _linear_terms(data, settings)
    scores = {
        feature: models[feature].zscore(age, linear, harmonized[:, index])
        for index, feature in enumerate(features)
    }
    return pd.DataFrame(scores, index=data.index)


def cross_validated_reference_scores(
    reference: pd.DataFrame, features: list[str], settings: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold control deviations; harmonization is re-learned per fold."""
    folds = StratifiedKFold(
        n_splits=int(settings.get("cv_folds", 10)), shuffle=True, random_state=seed
    )
    scores = pd.DataFrame(index=reference.index, columns=features, dtype=float)
    harmonized_test = pd.DataFrame(index=reference.index, columns=features, dtype=float)
    predicted_mean = pd.DataFrame(index=reference.index, columns=features, dtype=float)
    age_center = float(reference["age"].mean())
    for train_index, test_index in folds.split(reference, reference["scanner_epoch"]):
        train = reference.iloc[train_index]
        test = reference.iloc[test_index]
        combat = _combat(settings, age_center)
        train_values = combat.fit_transform(train, features)
        test_values = combat.transform(test, features)
        models = _fit_models(train, train_values, features, settings)
        scores.loc[test.index] = _score(test, test_values, models, features, settings).to_numpy()
        harmonized_test.loc[test.index] = test_values
        for feature in features:
            mean, _ = models[feature].predict(
                test["age"].to_numpy(dtype=float), _linear_terms(test, settings)
            )
            predicted_mean.loc[test.index, feature] = mean
    threshold = float(settings.get("extreme_threshold", 2.6))
    rows = []
    for feature in features:
        z = scores[feature].to_numpy(dtype=float)
        observed = harmonized_test[feature].to_numpy(dtype=float)
        error = observed - predicted_mean[feature].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "healthy_n": int(np.isfinite(z).sum()),
                "explained_variance": float(1 - error.var() / observed.var()),
                "smse": float(np.mean(error**2) / observed.var()),
                "z_mean": float(np.mean(z)),
                "z_sd": float(np.std(z, ddof=1)),
                "z_skew": float(stats.skew(z)),
                "z_excess_kurtosis": float(stats.kurtosis(z)),
                "ks_statistic": float(stats.kstest(z, "norm").statistic),
                "extreme_fraction": float(np.mean(np.abs(z) > threshold)),
                "expected_extreme_fraction": float(2 * stats.norm.sf(threshold)),
            }
        )
    return scores, pd.DataFrame(rows)


def _eta_squared_kruskal(values: np.ndarray, groups: np.ndarray) -> float:
    labels = [values[groups == label] for label in np.unique(groups)]
    labels = [group for group in labels if len(group) > 1]
    if len(labels) < 2:
        return float("nan")
    statistic = stats.kruskal(*labels).statistic
    return float((statistic - len(labels) + 1) / (len(values) - len(labels)))


def confound_checks(
    reference: pd.DataFrame, raw: pd.DataFrame, scores: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Age, motion and scanner-epoch effects in controls before and after modelling."""
    epochs = reference["scanner_epoch"].astype(str).to_numpy()
    sites = reference["site"].astype(str).to_numpy()
    age = reference["age"].to_numpy(dtype=float)
    motion = reference["mean_fd"].to_numpy(dtype=float) if "mean_fd" in reference else None
    rows = []
    for feature in features:
        before = raw.loc[reference.index, feature].to_numpy(dtype=float)
        after = scores.loc[reference.index, feature].to_numpy(dtype=float)
        before_within_site = before.copy()
        for site in np.unique(sites):
            mask = sites == site
            before_within_site[mask] -= np.median(before[mask])
        row = {
            "feature": feature,
            "age_spearman_input": float(stats.spearmanr(before, age).statistic),
            "age_spearman_deviation": float(stats.spearmanr(after, age).statistic),
            "epoch_eta2_input": _eta_squared_kruskal(before, epochs),
            "epoch_eta2_input_within_site": _eta_squared_kruskal(before_within_site, epochs),
            "epoch_eta2_deviation": _eta_squared_kruskal(after, epochs),
        }
        if motion is not None:
            row["motion_spearman_input"] = float(stats.spearmanr(before, motion).statistic)
            row["motion_spearman_deviation"] = float(stats.spearmanr(after, motion).statistic)
        rows.append(row)
    return pd.DataFrame(rows)


def _fdr(p_values: pd.Series) -> np.ndarray:
    finite = p_values.notna().to_numpy()
    adjusted = np.full(len(p_values), np.nan)
    if finite.any():
        adjusted[finite] = multipletests(p_values[finite], method="fdr_bh")[1]
    return adjusted


def deviation_burden(scores: pd.DataFrame, features: list[str], threshold: float) -> pd.DataFrame:
    values = scores[features].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "n_extreme_positive": (values > threshold).sum(axis=1),
            "n_extreme_negative": (values < -threshold).sum(axis=1),
            "n_extreme": (np.abs(values) > threshold).sum(axis=1),
            "mean_squared_deviation": np.mean(values**2, axis=1),
        },
        index=scores.index,
    )


def extreme_deviation_frequencies(
    scores: pd.DataFrame,
    roles: pd.Series,
    groups: dict[str, pd.Series],
    features: list[str],
    threshold: float,
) -> pd.DataFrame:
    """Per feature and direction: extreme-deviation rate in each group vs controls."""
    reference = scores.loc[roles == "healthy_reference", features]
    rows = []
    for group, mask in groups.items():
        clinical = scores.loc[mask, features]
        if clinical.empty:
            continue
        for feature in features:
            for direction, sign in (("positive", 1.0), ("negative", -1.0)):
                in_group = int((sign * clinical[feature] > threshold).sum())
                in_reference = int((sign * reference[feature] > threshold).sum())
                table = [
                    [in_group, len(clinical) - in_group],
                    [in_reference, len(reference) - in_reference],
                ]
                odds, p_value = stats.fisher_exact(table)
                rows.append(
                    {
                        "group": group,
                        "feature": feature,
                        "direction": direction,
                        "group_n": len(clinical),
                        "group_extreme_n": in_group,
                        "group_extreme_pct": 100 * in_group / len(clinical),
                        "healthy_n": len(reference),
                        "healthy_extreme_pct": 100 * in_reference / len(reference),
                        "odds_ratio": float(odds),
                        "p_value": float(p_value),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_fdr"] = result.groupby("group")["p_value"].transform(
            lambda values: pd.Series(_fdr(values), index=values.index)
        )
    return result


def burden_comparisons(
    burden: pd.DataFrame, roles: pd.Series, groups: dict[str, pd.Series], threshold: float
) -> pd.DataFrame:
    reference = burden.loc[roles == "healthy_reference"]
    rows = []
    for group, mask in groups.items():
        clinical = burden.loc[mask]
        if clinical.empty:
            continue
        for measure in ("n_extreme", "mean_squared_deviation"):
            test = stats.mannwhitneyu(clinical[measure], reference[measure])
            rows.append(
                {
                    "group": group,
                    "measure": measure,
                    "group_n": len(clinical),
                    "group_median": float(clinical[measure].median()),
                    "group_mean": float(clinical[measure].mean()),
                    "healthy_median": float(reference[measure].median()),
                    "healthy_mean": float(reference[measure].mean()),
                    "rank_biserial": float(
                        2 * test.statistic / (len(clinical) * len(reference)) - 1
                    ),
                    "p_value": float(test.pvalue),
                }
            )
        rows.append(
            {
                "group": group,
                "measure": f"any_extreme_abs_z_gt_{threshold:g}_pct",
                "group_n": len(clinical),
                "group_mean": float(100 * (clinical["n_extreme"] > 0).mean()),
                "healthy_mean": float(100 * (reference["n_extreme"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def overlap_summary(
    scores: pd.DataFrame,
    roles: pd.Series,
    groups: dict[str, pd.Series],
    features: list[str],
    threshold: float,
) -> pd.DataFrame:
    """How much do extreme deviations coincide across individuals?"""
    rows = []
    for group, mask in {"healthy_reference": roles == "healthy_reference", **groups}.items():
        values = scores.loc[mask, features]
        if values.empty:
            continue
        positive = (values > threshold).mean()
        negative = (values < -threshold).mean()
        shared = pd.concat([positive.add_suffix("+"), negative.add_suffix("-")])
        rows.append(
            {
                "group": group,
                "n": len(values),
                "pct_with_any_extreme": float(100 * (values.abs() > threshold).any(axis=1).mean()),
                "max_pct_sharing_one_extreme": float(100 * shared.max()),
                "most_shared_extreme": str(shared.idxmax()),
                "median_pct_sharing_one_extreme": float(100 * shared.median()),
            }
        )
    return pd.DataFrame(rows)


def clinical_associations(
    scores: pd.DataFrame,
    burden: pd.DataFrame,
    cohort: pd.DataFrame,
    mask: pd.Series,
    features: list[str],
    measures: list[str],
) -> pd.DataFrame:
    """Spearman associations of deviation burden and feature deviations with measures."""
    predictors = {
        "mean_squared_deviation": burden["mean_squared_deviation"],
        "n_extreme": burden["n_extreme"],
        **{feature: scores[feature] for feature in features},
    }
    rows = []
    for measure in measures:
        outcome = cohort.loc[mask, measure]
        for name, predictor in predictors.items():
            pair = pd.concat([predictor.loc[mask], outcome], axis=1).dropna()
            if len(pair) < 20:
                continue
            result = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            rows.append(
                {
                    "measure": measure,
                    "predictor": name,
                    "predictor_type": "burden" if name in burden else "feature",
                    "n": len(pair),
                    "spearman_rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_fdr"] = _fdr(table["p_value"])
    return table


def age_curves(
    models: dict[str, NormativeFeatureModel],
    ages: np.ndarray,
    other_linear: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Harmonized-scale reference centiles for plotting normative trajectories.

    Covariates other than sex (e.g. log mean FD) are held at `other_linear`.
    """
    other_linear = other_linear or {}
    rows = []
    for feature, model in models.items():
        for female in (0.0, 1.0):
            linear = np.column_stack(
                [np.full(len(ages), value) for value in (female, *other_linear.values())]
            )
            mean, sd = model.predict(ages, linear)
            for centile in (5, 25, 50, 75, 95):
                standardized = stats.norm.ppf(centile / 100)
                location, log_scale, skew, log_tail = model.shape
                # invert z = sinh(tail * asinh((r - loc) / scale) - skew)
                residual = location + np.exp(log_scale) * np.sinh(
                    (np.arcsinh(standardized) + skew) / np.exp(log_tail)
                )
                rows.extend(
                    {
                        "feature": feature,
                        "female": int(female),
                        **other_linear,
                        "age": float(age),
                        "centile": centile,
                        "value": float(value),
                    }
                    for age, value in zip(ages, mean + residual * sd, strict=True)
                )
    return pd.DataFrame(rows)


def run_normative_pipeline(config: Config) -> dict[str, Any]:
    root = config.require_path("outputs.root")
    settings = config.get("normative", {}) or {}
    seed = int(config.get("project.random_seed", 2107))
    threshold = float(settings.get("extreme_threshold", 2.6))
    feature_path = config.require_path("inputs.feature_table")
    with stage("Load features and normative cohort"):
        features_table = pd.read_csv(feature_path, sep="\t")
        features = [column for column in features_table if column not in NON_FEATURE_COLUMNS]
        cohort, exclusions = read_normative_cohort(config)
        imaged = set(features_table.subject_id)
        data = cohort.merge(features_table[["subject_id", *features]], on="subject_id")
        incomplete = data[features].isna().any(axis=1)
        unknown = imaged - set(cohort.subject_id) - set(exclusions.subject_id)
        # Audit only participants with imaging features.
        exclusions = pd.concat(
            [
                exclusions.loc[exclusions.subject_id.isin(imaged)],
                pd.DataFrame({"subject_id": sorted(unknown), "reason": "not-in-clinical-database"}),
                pd.DataFrame(
                    {"subject_id": data.loc[incomplete, "subject_id"], "reason": "feature-missing"}
                ),
            ],
            ignore_index=True,
        )
        data = data.loc[~incomplete].reset_index(drop=True)
        motion_settings = config.get("motion", {}) or {}
        motion_columns: list[str] = []
        motion_audit = pd.DataFrame()
        if motion_settings:
            tasks = list(motion_settings.get("tasks", ["AP", "FM"]))
            motion = task_motion_table(
                config.require_path("inputs.task_data_root"),
                data.subject_id,
                str(config.get("selection.session", "ses-01")),
                tasks,
                float(motion_settings.get("fd_threshold_mm", 0.5)),
            )
            data = data.merge(motion, on="subject_id", how="left")
            reasons = motion_exclusion_reasons(
                data,
                tasks,
                float(motion_settings.get("max_mean_fd_mm", 0.5)),
                float(motion_settings.get("max_high_motion_pct", 20)),
            )
            if not motion_settings.get("exclude", True):
                # without exclusion, only participants lacking parameters are dropped
                reasons = reasons.where(reasons.str.startswith("motion-parameters-missing"), "")
            motion_audit = data[
                ["subject_id", "analysis_role", "diagnosis_group", *motion.columns[1:]]
            ].assign(motion_exclusion=reasons)
            exclusions = pd.concat(
                [
                    exclusions,
                    data.loc[reasons != "", ["subject_id", "diagnosis_group"]].assign(
                        reason=reasons[reasons != ""]
                    ),
                ],
                ignore_index=True,
            )
            data = data.loc[reasons == ""].reset_index(drop=True)
            data["log_mean_fd"] = np.log(data["mean_fd"])
            motion_columns = ["mean_fd", "log_mean_fd"]
        missing = [name for name in _linear_names(settings) if name not in data]
        if missing:
            raise ConfigError(f"normative.linear_covariates not available: {missing}")
        covariate_columns = [*COVARIATE_COLUMNS, *motion_columns]
        epoch_counts = data.groupby(["scanner_epoch", "analysis_role"]).size().unstack(fill_value=0)
        minimum = int(settings.get("min_batch_reference_n", 20))
        if (
            "healthy_reference" not in epoch_counts
            or (epoch_counts["healthy_reference"] < minimum).any()
        ):
            raise ValueError(
                f"Every scanner epoch needs at least {minimum} healthy controls:\n{epoch_counts}"
            )
        report(
            f"Normative input: {int((data.analysis_role == 'healthy_reference').sum())} controls, "
            f"{int((data.analysis_role == 'clinical').sum())} patients, {len(features)} features, "
            f"{len(epoch_counts)} scanner epochs"
        )
    reference = data.loc[data.analysis_role == "healthy_reference"]
    clinical = data.loc[data.analysis_role == "clinical"]
    with stage("Cross-validated normative models in healthy controls"):
        reference_scores, calibration = cross_validated_reference_scores(
            reference, features, settings, seed
        )
    with stage("Fit final normative models and score patients"):
        combat = _combat(settings, float(reference["age"].mean()))
        reference_values = combat.fit_transform(reference, features)
        models = _fit_models(reference, reference_values, features, settings)
        clinical_scores = _score(
            clinical, combat.transform(clinical, features), models, features, settings
        )
        scores = pd.concat([reference_scores, clinical_scores]).loc[data.index]
    with stage("Summarize deviations"):
        groups = {
            name: (data.analysis_role == "clinical") & data.diagnosis_group.isin(members)
            for name, members in (settings.get("comparison_groups", {}) or {}).items()
        }
        burden = deviation_burden(scores, features, threshold)
        confounds = confound_checks(reference, data, scores, features)
        frequencies = extreme_deviation_frequencies(
            scores, data.analysis_role, groups, features, threshold
        )
        comparisons = burden_comparisons(burden, data.analysis_role, groups, threshold)
        overlap = overlap_summary(scores, data.analysis_role, groups, features, threshold)
        association_group = settings.get("association_group")
        associations = pd.DataFrame()
        if association_group in groups:
            associations = clinical_associations(
                scores,
                burden,
                data,
                groups[association_group],
                features,
                list((settings.get("clinical_measures") or {}).keys()),
            )
        shape = pd.DataFrame(
            {
                "feature": features,
                "shash_skew": [float(models[name].shape[2]) for name in features],
                "shash_tail_weight": [float(np.exp(models[name].shape[3])) for name in features],
            }
        )
        calibration = calibration.merge(shape, on="feature")
    with stage("Write normative outputs"):
        cohort_dir = root / "cohort"
        model_dir = root / "model"
        deviation_dir = root / "deviations"
        analysis_dir = root / "analysis"
        for directory in (cohort_dir, model_dir, deviation_dir, analysis_dir):
            directory.mkdir(parents=True, exist_ok=True)
        data[covariate_columns].to_csv(cohort_dir / "participants.tsv", sep="\t", index=False)
        exclusions.to_csv(cohort_dir / "exclusions.tsv", sep="\t", index=False)
        epoch_counts.to_csv(cohort_dir / "scanner_epoch_counts.tsv", sep="\t")
        if not motion_audit.empty:
            motion_audit.to_csv(cohort_dir / "motion.tsv", sep="\t", index=False)
        pd.concat([data[covariate_columns], scores], axis=1).to_csv(
            deviation_dir / "deviation_scores.tsv", sep="\t", index=False
        )
        pd.concat(
            [data[["subject_id", "analysis_role", "diagnosis_group"]], burden], axis=1
        ).to_csv(deviation_dir / "deviation_burden.tsv", sep="\t", index=False)
        calibration.to_csv(model_dir / "calibration.tsv", sep="\t", index=False)
        confounds.to_csv(model_dir / "confound_checks.tsv", sep="\t", index=False)
        held = {name: float(reference[name].median()) for name in _linear_names(settings)[1:]}
        age_curves(models, np.arange(18, 66, 1.0), held).to_csv(
            model_dir / "reference_centiles.tsv", sep="\t", index=False
        )
        frequencies.to_csv(
            analysis_dir / "extreme_deviation_frequencies.tsv", sep="\t", index=False
        )
        comparisons.to_csv(analysis_dir / "burden_comparisons.tsv", sep="\t", index=False)
        overlap.to_csv(analysis_dir / "overlap.tsv", sep="\t", index=False)
        associations.to_csv(analysis_dir / "clinical_associations.tsv", sep="\t", index=False)
    status: dict[str, Any] = {
        "mode": "normative-deviation",
        "feature_table": str(feature_path),
        "features": features,
        "healthy_n": len(reference),
        "clinical_n": len(clinical),
        "excluded_n": len(exclusions),
        "scanner_epochs": {
            str(epoch): {str(role): int(n) for role, n in counts.items()}
            for epoch, counts in epoch_counts.iterrows()
        },
        "comparison_groups": {name: int(mask.sum()) for name, mask in groups.items()},
        "extreme_threshold": threshold,
        "calibration_summary": {
            "median_explained_variance": float(calibration.explained_variance.median()),
            "max_abs_z_mean": float(calibration.z_mean.abs().max()),
            "z_sd_range": [float(calibration.z_sd.min()), float(calibration.z_sd.max())],
            "healthy_extreme_fraction_range": [
                float(calibration.extreme_fraction.min()),
                float(calibration.extreme_fraction.max()),
            ],
            "max_abs_age_spearman_deviation": float(confounds.age_spearman_deviation.abs().max()),
            "max_epoch_eta2_deviation": float(confounds.epoch_eta2_deviation.max()),
        },
        "linear_covariates": _linear_names(settings),
    }
    if motion_columns:
        healthy = data.analysis_role == "healthy_reference"
        status["motion"] = {
            "settings": motion_settings,
            "excluded_by_group": {
                str(group): {
                    "excluded": int((rows.motion_exclusion != "").sum()),
                    "assessed": len(rows),
                }
                for group, rows in motion_audit.assign(
                    group=motion_audit.diagnosis_group.where(
                        motion_audit.analysis_role == "clinical", "HC"
                    )
                ).groupby("group")
            },
            "max_abs_motion_spearman_input": float(confounds.motion_spearman_input.abs().max()),
            "max_abs_motion_spearman_deviation": float(
                confounds.motion_spearman_deviation.abs().max()
            ),
            "burden_motion_spearman_healthy": float(
                stats.spearmanr(
                    burden.loc[healthy, "mean_squared_deviation"], data.loc[healthy, "mean_fd"]
                ).statistic
            ),
        }
    clinical_inputs = [Path(value) for value in config.get("inputs.clinical_files", [])]
    write_provenance(root / "provenance.json", config.path, [*clinical_inputs, feature_path])
    (root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status
