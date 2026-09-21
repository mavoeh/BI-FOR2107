import numpy as np
import pandas as pd

from for2107_repro.clustering import (
    empirical_p_value,
    evaluate_solutions,
    fit_clusters,
    leave_one_out_stability,
    null_silhouette_tests,
    repeated_split_half_replication,
    split_half_replication,
    subsample_stability,
)


def test_empirical_p_value_never_returns_zero() -> None:
    assert empirical_p_value(0, 10_000) == 1 / 10_001
    assert empirical_p_value(10_000, 10_000) == 1.0


def synthetic_profiles(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    profile_a = np.array([2, 1, 0, -1, -2, 0.5], dtype=float)
    profile_b = -profile_a
    return np.vstack(
        [profile_a + rng.normal(scale=0.12, size=6) for _ in range(15)]
        + [profile_b + rng.normal(scale=0.12, size=6) for _ in range(15)]
    )


def test_correlation_clustering_recovers_profiles() -> None:
    values = synthetic_profiles()
    labels = fit_clusters(values, 2).labels
    assert len(set(labels[:15])) == 1
    assert len(set(labels[15:])) == 1
    assert labels[0] != labels[-1]


def test_validation_metrics_execute_deterministically() -> None:
    values = synthetic_profiles()
    evaluation = evaluate_solutions(values, [2, 3])
    assert evaluation.loc[evaluation["n_clusters"] == 2, "silhouette"].iloc[0] > 0.8
    nulls = null_silhouette_tests(values, [2], simulations=3, permutations=3, seed=4)
    assert list(nulls["n_clusters"]) == [2]
    assert len(leave_one_out_stability(values, [2])) == 1
    assert len(subsample_stability(values, [2], iterations=3, seed=4)) == 1
    split = split_half_replication(values, 2, seed=4)
    assert split["mean_profile_correlation"] > 0.9


def test_parallel_validation_is_numerically_identical() -> None:
    values = synthetic_profiles()
    serial_nulls = null_silhouette_tests(
        values, [2, 3], simulations=8, permutations=8, seed=41, n_jobs=1
    )
    parallel_nulls = null_silhouette_tests(
        values, [2, 3], simulations=8, permutations=8, seed=41, n_jobs=3
    )
    pd.testing.assert_frame_equal(serial_nulls, parallel_nulls, check_exact=True)

    serial_loo = leave_one_out_stability(values, [2, 3], n_jobs=1)
    parallel_loo = leave_one_out_stability(values, [2, 3], n_jobs=3)
    pd.testing.assert_frame_equal(serial_loo, parallel_loo, check_exact=True)

    serial_subsample = subsample_stability(values, [2, 3], iterations=12, seed=41, n_jobs=1)
    parallel_subsample = subsample_stability(values, [2, 3], iterations=12, seed=41, n_jobs=3)
    pd.testing.assert_frame_equal(serial_subsample, parallel_subsample, check_exact=True)

    serial_split = repeated_split_half_replication(values, 2, iterations=6, seed=41, n_jobs=1)
    parallel_split = repeated_split_half_replication(values, 2, iterations=6, seed=41, n_jobs=3)
    pd.testing.assert_frame_equal(serial_split, parallel_split, check_exact=True)
