from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass(frozen=True)
class ImputationAudit:
    feature: str
    batch: str
    missing: int
    method: str


def random_forest_pmm(
    data: pd.DataFrame,
    features: list[str],
    batch_column: str,
    *,
    seed: int = 2107,
    donors: int = 5,
    iterations: int = 1,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random-forest predictive mean matching, stratified by scanner batch."""
    result = data.copy()
    audits: list[ImputationAudit] = []
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        for batch, index in result.groupby(batch_column, dropna=False).groups.items():
            group = result.loc[index, features]
            for feature in features:
                missing_mask = group[feature].isna()
                n_missing = int(missing_mask.sum())
                if not n_missing:
                    continue
                observed_mask = ~missing_mask
                observed = group.loc[observed_mask, feature]
                if len(observed) < max(5, donors):
                    fill = observed.median()
                    result.loc[group.index[missing_mask], feature] = fill
                    audits.append(ImputationAudit(feature, str(batch), n_missing, "within-batch-median"))
                    continue
                predictors = [column for column in features if column != feature]
                x = group[predictors].copy()
                x = x.fillna(x.median()).fillna(0)
                forest = RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=3,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                    n_jobs=n_jobs,
                )
                forest.fit(x.loc[observed_mask], observed)
                predicted_observed = forest.predict(x.loc[observed_mask])
                predicted_missing = forest.predict(x.loc[missing_mask])
                observed_values = observed.to_numpy()
                imputed = []
                for prediction in predicted_missing:
                    nearest = np.argsort(np.abs(predicted_observed - prediction))[:donors]
                    imputed.append(observed_values[rng.choice(nearest)])
                result.loc[group.index[missing_mask], feature] = imputed
                audits.append(ImputationAudit(feature, str(batch), n_missing, "random-forest-pmm"))
    return result, pd.DataFrame(item.__dict__ for item in audits)


def combat_harmonize(
    data: pd.DataFrame,
    features: list[str],
    batch_column: str,
    covariates: Iterable[str] = (),
) -> pd.DataFrame:
    # ComBat has no batch effect to estimate for a single scanner/site and
    # neuroHarmonize is not defined for that degenerate design.
    if data[batch_column].nunique(dropna=False) <= 1:
        return data.copy()
    try:
        from neuroHarmonize import harmonizationLearn
    except ImportError as exc:
        raise RuntimeError("Install neuroHarmonize to run ComBat; no silent fallback is used") from exc
    covariate_names = [name for name in covariates if name in data]
    covars = data[[batch_column, *covariate_names]].copy().rename(columns={batch_column: "SITE"})
    _, adjusted = harmonizationLearn(data[features].to_numpy(dtype=float), covars)
    result = data.copy()
    result.loc[:, features] = adjusted
    return result


def healthy_reference_scores(
    data: pd.DataFrame,
    features: list[str],
    healthy_mask: pd.Series | np.ndarray,
    *,
    minimum_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    healthy = data.loc[np.asarray(healthy_mask, dtype=bool), features]
    if len(healthy) < minimum_n:
        raise ValueError(f"Healthy reference n={len(healthy)} is below required n={minimum_n}")
    means = healthy.mean()
    standard_deviations = healthy.std(ddof=1)
    invalid = standard_deviations[(standard_deviations <= 0) | standard_deviations.isna()]
    if not invalid.empty:
        raise ValueError(f"Invalid healthy-reference standard deviation: {list(invalid.index)}")
    scores = data.copy()
    scores.loc[:, features] = (data[features] - means) / standard_deviations
    audit = pd.DataFrame(
        {
            "feature": features,
            "healthy_n": healthy.notna().sum().to_numpy(),
            "mean": means.to_numpy(),
            "sd": standard_deviations.to_numpy(),
        }
    )
    return scores, audit
