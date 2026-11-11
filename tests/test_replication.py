import numpy as np
import pandas as pd
import pytest

from for2107_repro.replication import (
    _icc_a1,
    circuit_of,
    clinical_test_calibration,
    compare_clusters,
    faithful_scores,
    mask_high_motion,
    match_labels,
    measurement_agreement,
    tozzi_name,
)


def test_circuit_of_maps_feature_names_to_tozzi_circuits() -> None:
    assert circuit_of("D2D1_default_ampfc_ag_l_connectivity") == "D_C"
    assert circuit_of("A5A7_attention_aipl_r_pcun_r_connectivity") == "A_C"
    assert circuit_of("FM_faces_vs_shapes_Negativethreat_dACC_M_activation_adapted") == "NTC_A"
    assert (
        circuit_of("AP_sad_vs_neutral_Negativesad_AI_L__Negativesad_pgACC_M_ppi_adapted") == "NS_C"
    )
    assert circuit_of("AP_happy_vs_neutral_Positive_VS_L_activation_adapted") == "P_A"
    with pytest.raises(ValueError):
        circuit_of("something_else")


def test_motion_mask_pools_frames_over_the_runs_of_a_group() -> None:
    data = pd.DataFrame({"subject_id": ["s1", "s2"], "edge": [0.1, 0.2], "ap": [1.0, 2.0]})
    motion = pd.DataFrame(
        {
            "subject_id": ["s1", "s2"],
            "high_motion_pct_AP": [40.0, 10.0],
            "n_frames_AP": [100, 100],
            "high_motion_pct_FM": [0.0, 10.0],
            "n_frames_FM": [100, 100],
        }
    )
    groups = {"task_free": ["edge"], "AP": ["ap"]}
    masked, audit = mask_high_motion(
        data, groups, {"task_free": ["AP", "FM"], "AP": ["AP"]}, motion, 0.25
    )
    # s1: 40% in AP alone, 20% pooled over AP + FM
    assert masked.loc[0, "edge"] == 0.1 and np.isnan(masked.loc[0, "ap"])
    assert audit.loc[0, "high_motion_fraction_task_free"] == pytest.approx(0.2)
    assert masked.loc[1].notna().all()


def test_ppi_directions_are_averaged_after_standardization() -> None:
    rng = np.random.default_rng(0)
    n = 60
    data = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(n)],
            "site": ["a"] * n,
            "x__phys_A": rng.normal(0, 1, n),
            "x__phys_B": rng.normal(5, 10, n),
            "edge": rng.normal(size=n),
        }
    )
    healthy = pd.Series(np.arange(n) < 30)
    scores, final, _ = faithful_scores(
        data, ["edge", "x__phys_A", "x__phys_B"], "site", healthy, seed=1
    )
    assert final == ["edge", "x"]
    expected = (scores["x__phys_A"] + scores["x__phys_B"]) / 2
    np.testing.assert_allclose(scores["x"], expected)
    # each direction is on the control scale, so neither dominates the average
    assert scores.loc[healthy, "x__phys_B"].std() == pytest.approx(1.0)


def test_tozzi_name_uses_half_sd_threshold() -> None:
    assert tozzi_name(pd.Series({"D_C": 0.6, "S_C": 0.2, "A_C": -0.7})) == "D_C+ A_C-"
    assert tozzi_name(pd.Series({"D_C": 0.1})) == "none >= 0.5 SD"


def test_match_labels_follows_centroids_not_label_numbers() -> None:
    rng = np.random.default_rng(2)
    profile_a, profile_b = np.array([1.0, -1.0, 0.5, 0.0]), np.array([-1.0, 1.0, 0.0, 0.5])
    matrix = np.vstack(
        [profile_a + rng.normal(0, 0.1, (20, 4)), profile_b + rng.normal(0, 0.1, (20, 4))]
    )
    reference_labels = np.repeat([1, 2], 20)
    swapped = np.repeat([2, 1], 20)
    np.testing.assert_array_equal(
        match_labels(matrix, reference_labels, matrix, swapped), reference_labels
    )


def test_compare_clusters_one_sample_wilcoxon_direction() -> None:
    values = pd.DataFrame({"m": np.r_[np.arange(10, 30), np.arange(0, 20)].astype(float)})
    labels = np.repeat([1, 2], 20)
    table = compare_clusters(values, labels, ["m"]).set_index("cluster")
    assert table.loc[1, "z"] > 0 > table.loc[2, "z"]
    assert table.loc[1, "median_other"] == pytest.approx(9.5)


def test_measurement_agreement_is_perfect_for_identical_measurements() -> None:
    rng = np.random.default_rng(3)
    a = pd.DataFrame(rng.normal(size=(80, 6)), columns=list("abcdef"))
    features, assignment, summary = measurement_agreement(a, a.copy(), [2, 3], seed=4)
    assert summary["median_profile_correlation"] == pytest.approx(1.0)
    assert (features.icc_a1 > 0.999).all()
    assert (assignment.kappa_retest == 1.0).all()
    assert (assignment.ari_independent_clusterings > 0.99).all()


def test_icc_a1_penalizes_systematic_offsets() -> None:
    x = np.random.default_rng(5).normal(size=200)
    assert _icc_a1(x, x) == pytest.approx(1.0)
    # ICC(A,1) = var / (var + offset² / 2) for a constant offset
    assert _icc_a1(x, x + 2.0) == pytest.approx(x.var() / (x.var() + 2.0), abs=0.02)


def test_tozzi_wilcoxon_is_biased_for_skewed_scales_but_mann_whitney_is_not() -> None:
    rng = np.random.default_rng(6)
    # zero-inflated, right-skewed item (like HAMD suicide): no true group difference
    values = pd.DataFrame({"item": rng.choice([0, 0, 0, 0, 1, 2, 3], size=600).astype(float)})
    labels = rng.choice([1, 2, 3], size=600)
    calibration = clinical_test_calibration(values, labels, ["item"], seed=7, iterations=50)
    rates = calibration.set_index("test").false_positive_rate
    assert rates["tozzi_wilcoxon"] > 0.5
    assert rates["mann_whitney"] < 0.15


def test_mann_whitney_effect_is_rank_biserial() -> None:
    values = pd.DataFrame({"m": np.r_[np.full(10, 2.0), np.full(10, 1.0)]})
    table = compare_clusters(values, np.repeat([1, 2], 10), ["m"]).set_index("cluster")
    assert table.loc[1, "r"] == pytest.approx(1.0)
    assert table.loc[2, "r"] == pytest.approx(-1.0)
