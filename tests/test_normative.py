import numpy as np
import pandas as pd
import pytest
from scipy import stats

from for2107_repro.config import ConfigError
from for2107_repro.normative import (
    NormativeFeatureModel,
    assign_scanner_epochs,
    burden_comparisons,
    cross_validated_reference_scores,
    deviation_burden,
    extreme_deviation_frequencies,
    fit_shash,
)

RULES = {
    "A_pre": {"Site": 1, "Upgrade": 0},
    "A_post": {"Site": 1, "Upgrade": 1},
    "B": {"Site": 2},
}


def test_scanner_epochs_follow_rules_and_leave_unmatched_missing() -> None:
    codes = pd.DataFrame({"Site": [1, 1, 2, np.nan, 1], "Upgrade": [0, 1, 0, 0, -99]})
    labels = assign_scanner_epochs(codes, RULES)
    assert labels.iloc[:3].tolist() == ["A_pre", "A_post", "B"]
    assert labels.iloc[3:].isna().all()


def test_overlapping_scanner_epoch_rules_are_rejected() -> None:
    codes = pd.DataFrame({"Site": [1], "Upgrade": [0]})
    with pytest.raises(ConfigError):
        assign_scanner_epochs(codes, {**RULES, "A_any": {"Site": 1}})


def test_shash_fit_is_near_identity_for_gaussian_data() -> None:
    values = np.random.default_rng(1).standard_normal(4000)
    location, log_scale, skew, log_tail = fit_shash(values)
    assert abs(location) < 0.1 and abs(log_scale) < 0.1
    assert abs(skew) < 0.1 and abs(log_tail) < 0.1


def _simulated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 65, n)
    female = rng.integers(0, 2, n).astype(float)
    # nonlinear age trend, sex effect, age-dependent spread, heavy tails
    noise = stats.t.rvs(df=4, size=n, random_state=rng) * (0.5 + 0.02 * age)
    y = 2.0 + 0.002 * (age - 40) ** 2 - 0.03 * age + 0.4 * female + noise
    return age, female, y


def test_normative_model_calibrates_heavy_tailed_heteroscedastic_data() -> None:
    age, female, y = _simulated(3000, seed=2)
    model = NormativeFeatureModel().fit(age, female, y)
    test_age, test_female, test_y = _simulated(3000, seed=3)
    z = model.zscore(test_age, test_female, test_y)
    assert abs(z.mean()) < 0.06
    assert abs(z.std() - 1) < 0.06
    assert abs(stats.spearmanr(z, test_age).statistic) < 0.05
    assert abs(np.std(z[test_age > 50]) - np.std(z[test_age < 30])) < 0.12
    # tails are calibrated, unlike a Gaussian z on t(4) noise
    assert np.mean(np.abs(z) > 2.6) == pytest.approx(2 * stats.norm.sf(2.6), abs=0.006)


def test_cross_validated_scores_remove_scanner_epoch_shift() -> None:
    pytest.importorskip("neuroCombat")
    rng = np.random.default_rng(4)
    n = 400
    age, female, _ = _simulated(n, seed=5)
    epoch = np.where(np.arange(n) % 2 == 0, "pre", "post")
    shifted = rng.standard_normal((n, 2)) + np.where(epoch == "post", 3.0, 0.0)[:, None]
    reference = pd.DataFrame(
        {
            "age": age,
            "female": female,
            "scanner_epoch": epoch,
            "x": shifted[:, 0],
            "y": shifted[:, 1],
        }
    )
    scores, calibration = cross_validated_reference_scores(
        reference, ["x", "y"], {"cv_folds": 5}, seed=6
    )
    assert scores.notna().all().all()
    for feature in ("x", "y"):
        gap = (
            scores.loc[epoch == "post", feature].mean() - scores.loc[epoch == "pre", feature].mean()
        )
        assert abs(gap) < 0.3
    assert set(calibration.feature) == {"x", "y"}


def test_burden_and_extreme_frequencies() -> None:
    scores = pd.DataFrame({"a": [0.0, 3.0, -3.0, 0.1], "b": [2.7, 0.0, 0.0, 0.0]})
    burden = deviation_burden(scores, ["a", "b"], 2.6)
    assert burden.n_extreme.tolist() == [1, 1, 1, 0]
    assert burden.n_extreme_negative.tolist() == [0, 0, 1, 0]
    roles = pd.Series(["healthy_reference", "clinical", "clinical", "healthy_reference"])
    groups = {"patients": roles == "clinical"}
    table = extreme_deviation_frequencies(scores, roles, groups, ["a", "b"], 2.6)
    row = table.set_index(["feature", "direction"]).loc[("a", "positive")]
    assert row.group_extreme_n == 1 and row.healthy_extreme_pct == 0
    comparisons = burden_comparisons(burden, roles, groups, 2.6)
    any_row = comparisons.loc[comparisons.measure.str.startswith("any_extreme")].iloc[0]
    assert any_row.group_mean == 100 and any_row.healthy_mean == 50


def test_extra_linear_covariate_removes_its_effect() -> None:
    age, female, y = _simulated(3000, seed=7)
    rng = np.random.default_rng(8)
    motion = rng.normal(0, 1, len(age))
    y = y + 0.8 * motion
    model = NormativeFeatureModel().fit(age, np.column_stack([female, motion]), y)
    z = model.zscore(age, np.column_stack([female, motion]), y)
    assert abs(stats.spearmanr(z, motion).statistic) < 0.03
    sex_only = NormativeFeatureModel().fit(age, female, y).zscore(age, female, y)
    assert stats.spearmanr(sex_only, motion).statistic > 0.3
