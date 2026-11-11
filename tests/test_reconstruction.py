import numpy as np
import pandas as pd
import pytest

from for2107_repro.reconstruction import (
    EntropyBalance,
    assign_profiles,
    base_correlation,
    draw_pool,
    mixture_kl,
    weight_diagnostics,
)


def _balance(pool, labels, shares, profiles, cross=None):
    k, p = profiles.shape
    return EntropyBalance(
        pool, labels, shares, profiles, np.full(k, 1e-4), np.full((k, p), 1e-3), cross
    )


def test_reweighting_reaches_target_sizes_and_profiles() -> None:
    rng = np.random.default_rng(0)
    pool = rng.standard_normal((40_000, 4)).astype(np.float32)
    profiles = np.array([[0.8, 0.2, -0.3, 0.0], [-0.5, -0.4, 0.6, 0.1]])
    labels = assign_profiles(pool, profiles)
    shares = np.array([0.7, 0.3])
    weights, solver = _balance(pool, labels, shares, profiles).solve()
    assert solver["converged"]
    for label in range(2):
        members = labels == label
        assert weights[members].sum() == pytest.approx(shares[label], abs=1e-3)
        mean = weights[members] @ pool[members] / weights[members].sum()
        np.testing.assert_allclose(mean, profiles[label], atol=0.01)


def test_reweighting_leaves_equal_weights_when_targets_already_hold() -> None:
    rng = np.random.default_rng(1)
    pool = rng.standard_normal((20_000, 3)).astype(np.float32)
    labels = (pool[:, 0] > 0).astype(int)
    profiles = np.vstack([pool[labels == label].mean(axis=0) for label in (0, 1)])
    shares = np.bincount(labels) / len(labels)
    weights, _ = _balance(pool, labels, shares, profiles).solve()
    diagnostics = weight_diagnostics(weights)
    assert diagnostics["kl_from_equal"] < 1e-4
    assert diagnostics["effective_fraction"] > 0.999


def test_reweighting_matches_cross_block_correlations() -> None:
    rng = np.random.default_rng(2)
    pool = rng.standard_normal((40_000, 3)).astype(np.float32)
    labels = np.zeros(len(pool), dtype=int)
    profiles = np.zeros((1, 3))
    cross = (np.array([0]), np.array([1, 2]), np.array([[0.3, -0.2]]), np.full((1, 2), 1e-3))
    weights, _ = _balance(pool, labels, np.array([1.0]), profiles, cross).solve()
    moment = (pool[:, [0]] * weights[:, None]).T @ pool[:, [1, 2]]
    np.testing.assert_allclose(moment, [[0.3, -0.2]], atol=0.01)


def test_weight_diagnostics_for_equal_and_halved_pools() -> None:
    equal = np.full(1000, 1 / 1000)
    assert weight_diagnostics(equal)["effective_fraction"] == pytest.approx(1.0)
    half = np.r_[np.full(500, 1 / 500), np.zeros(500)]
    diagnostics = weight_diagnostics(half)
    assert diagnostics["effective_fraction"] == pytest.approx(0.5)
    assert diagnostics["kl_from_equal"] == pytest.approx(np.log(2))


def test_base_correlation_uses_published_cross_block() -> None:
    within = pd.DataFrame(np.eye(2), index=["D2D1", "NS1"], columns=["D2D1", "NS1"])
    cross = pd.DataFrame([[0.25]], index=pd.Index(["NS1"], name="task"), columns=["D2D1"])
    matrix = base_correlation(within, cross, ["D2D1", "NS1", "C1"])
    assert matrix[0, 1] == pytest.approx(0.25, abs=0.01)
    np.testing.assert_allclose(np.diag(matrix), 1.0)


def test_student_t_pool_has_unit_variance_and_heavy_tails() -> None:
    rng = np.random.default_rng(3)
    pool = draw_pool(200_000, np.zeros(2), np.eye(2), "student_t", rng, df=5)
    assert pool.var(axis=0) == pytest.approx([1, 1], abs=0.05)
    kurtosis = ((pool - pool.mean(axis=0)) ** 4).mean(axis=0) / pool.var(axis=0) ** 2
    assert (kurtosis > 4).all()


def test_mixture_kl_is_zero_without_subgroups_and_grows_with_separation() -> None:
    rng = np.random.default_rng(4)
    correlation = np.eye(3)
    same = mixture_kl(np.zeros((2, 3)), np.array([0.5, 0.5]), np.zeros(3), correlation, 20_000, rng)
    assert same == pytest.approx(0.0, abs=1e-9)
    apart = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    near = mixture_kl(0.5 * apart, np.array([0.5, 0.5]), np.zeros(3), correlation, 20_000, rng)
    far = mixture_kl(2 * apart, np.array([0.5, 0.5]), np.zeros(3), correlation, 20_000, rng)
    assert 0 < near < far
