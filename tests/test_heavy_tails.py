import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

from for2107_repro.clustering import _silhouette, fit_clusters
from for2107_repro.heavy_tails import (
    DEFAULT_DESIGN,
    battery,
    patient_values,
    score_sample,
    silhouette,
    simulate_sample,
    tozzi_like,
    within_biotype_correlation,
)


def test_fast_silhouette_matches_sklearn_including_singletons() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 5))
    square = squareform(pdist(x, metric="correlation"))
    labels = rng.integers(1, 5, 60)
    labels[0] = 9  # a singleton cluster scores 0
    assert silhouette(square, labels) == pytest.approx(
        silhouette_score(square, labels, metric="precomputed")
    )


def test_battery_silhouette_matches_tozzi_clustering() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((120, 8))
    settings = {
        "simulations": 3,
        "permutations": 3,
        "leave_one_out": 3,
        "subsample": 3,
        "split_half": 2,
    }
    by_k, summary = battery(x, [2, 6], settings, rng)
    assert by_k.set_index("k").loc[6, "silhouette"] == pytest.approx(
        _silhouette(fit_clusters(x, 6))
    )
    assert sum(summary["sizes_sorted"]) == 120


def test_whole_profile_tails_leave_correlation_distances_unchanged() -> None:
    rng = np.random.default_rng(2)
    z = rng.standard_normal((50, 6))
    scaled = z * rng.chisquare(3, size=(50, 1))
    np.testing.assert_allclose(pdist(z, "correlation"), pdist(scaled, "correlation"), atol=1e-12)


def test_feature_tails_keep_unit_variance_and_add_extremes() -> None:
    rng = np.random.default_rng(3)
    values = patient_values(100_000, np.eye(3), {"feature_t": 4}, rng)
    assert values.var(axis=0) == pytest.approx([1, 1, 1], abs=0.08)
    assert np.mean(np.abs(values) > 4) > 10 * 6.3e-5  # far above the normal rate


def test_simulated_sample_has_his_design_and_scores_patients() -> None:
    rng = np.random.default_rng(4)
    correlation = 0.3 * np.ones((5, 5)) + 0.7 * np.eye(5)
    sample = simulate_sample(correlation, {"study_factor": 0.5}, DEFAULT_DESIGN, rng)
    counts = sample.groupby(["analysis_role", "study"]).size()
    assert counts["clinical"].sum() == 801 and counts["healthy_reference"].sum() == 137
    assert counts[("clinical", "RAD")] == 309
    assert score_sample(sample).shape == (801, 5)


def test_tozzi_like_checks_each_published_property() -> None:
    by_k = pd.DataFrame(
        {
            "k": [5, 6, 7],
            "silhouette": [0.07, 0.065, 0.06],
            "p_multinormal": [0.01, 0.016, 0.02],
            "p_permuted": [0.0, 0.0, 0.0],
            "loo_ari": [0.8, 0.8, 0.8],
            "subsample_ari": [0.35, 0.35, 0.35],
        }
    )
    summary = {
        "largest_four_min_share": 0.19,
        "smallest_two_share": 0.07,
        "split_half_r": 0.91,
        "profile_mean_abs": 0.29,
        "share_negative_silhouette": 0.28,
    }
    checks = tozzi_like(by_k, summary, {})
    assert all(checks.values())
    by_k.loc[1, "p_multinormal"] = 0.3
    assert not tozzi_like(by_k, summary, {})["all"]


def test_biotypes_keep_the_total_correlation() -> None:
    rng = np.random.default_rng(5)
    correlation = 0.4 * np.ones((4, 4)) + 0.6 * np.eye(4)
    means = np.array([[0.5, 0.5, -0.3, 0.0], [-0.4, 0.1, 0.3, 0.2]])
    proportions = np.array([0.6, 0.4])
    values = patient_values(
        200_000, correlation, {"biotypes": 1.0}, rng, biotypes=(means, proportions)
    )
    np.testing.assert_allclose(np.cov(values, rowvar=False), correlation, atol=0.02)
    within = within_biotype_correlation(correlation, means, proportions)
    assert np.all(np.diag(within) < 1)


def test_patients_carry_the_mean_profile() -> None:
    rng = np.random.default_rng(6)
    mean = np.array([0.2, -0.1, 0.0])
    values = patient_values(50_000, np.eye(3), {}, rng, mean=mean)
    np.testing.assert_allclose(values.mean(axis=0), mean, atol=0.02)
