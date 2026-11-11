from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from for2107_repro.config import Config, ConfigError
from for2107_repro.stability import (
    SUMMARY_METRICS,
    build_references,
    compare_clusterings,
    covariance_factor,
    define_feature_sets,
    interpolate_scale,
    matched_comparison,
    noise_covariance,
    permutation_p,
    planted_keys,
    prepare_blocks,
    split_half,
    summarize_splits,
    tozzi_code_circuit,
    tozzi_correlation,
    tozzi_profile_match,
    tozzi_split_half_r,
    write_tozzi_checks,
)
from for2107_repro.stability_figures import read_table

SETTINGS = {"replicated": 0.8, "min_size": 10}


def _clustered(rng: np.random.Generator, n: int = 600, p: int = 20, spread: float = 3.0):
    centres = rng.normal(0, spread, (6, p))
    truth = rng.integers(0, 6, n)
    return centres[truth] + rng.normal(size=(n, p)), truth


def test_matched_comparison_puts_matches_on_the_diagonal_by_size() -> None:
    rng = np.random.default_rng(0)
    centroids = rng.normal(size=(6, 10))
    sizes = np.array([5, 60, 20, 40, 10, 30])
    order = rng.permutation(6)
    result = matched_comparison(centroids, sizes, centroids[order], sizes[order], **SETTINGS)
    np.testing.assert_allclose(np.diag(result["matrix"]), 1.0)
    np.testing.assert_array_equal(result["sizes_first"], np.sort(sizes)[::-1])
    assert result["mean_matched"] == pytest.approx(1.0)
    assert result["separation"] > 0.5
    # the five clusters with >= 10 members replicate; the cluster of 5 is too small
    assert result["n_replicated"] == 5


def test_split_half_separates_real_clusters_from_a_cluster_free_null() -> None:
    rng = np.random.default_rng(1)
    clustered, _ = _clustered(rng)
    real = split_half(clustered, 6, rng, **SETTINGS)
    null = split_half(rng.normal(size=clustered.shape), 6, rng, **SETTINGS)
    assert real["separation"] > 0.9 and real["ari_assignment"] > 0.9
    assert real["separation"] > null["separation"] + 0.3


def test_covariance_factor_rows_give_the_subset_covariance() -> None:
    rng = np.random.default_rng(2)
    covariance = np.cov(rng.normal(size=(200, 8)) @ rng.normal(size=(8, 8)), rowvar=False)
    factor = covariance_factor(covariance)
    idx = np.array([1, 4, 6])
    np.testing.assert_allclose(
        factor[idx] @ factor[idx].T, covariance[np.ix_(idx, idx)], atol=1e-10
    )


def test_planted_reference_places_biotype_means_on_tozzi_columns_only() -> None:
    rng = np.random.default_rng(3)
    template = pd.DataFrame({"D_C": [1.0, -1.0], "P_A": [0.5, 0.0]}, index=["b1", "b2"])
    circuits = np.array(["D_C", "", "P_A"])
    references = build_references(
        rng.normal(size=(300, 3)),
        rng.normal(size=(300, 3)),
        circuits,
        template,
        np.array([0.5, 0.5]),
        2.0,
    )
    np.testing.assert_allclose(references.planted_means, [[2.0, 0.0, 1.0], [-2.0, 0.0, 0.0]])
    planted = references.draw("planted", 4000, rng)
    assert planted[:, 0].std() > planted[:, 1].std() + 0.5
    assert references.subset(np.array([0, 2])).draw("null", 10, rng).shape == (10, 2)


def test_feature_sets_include_structured_and_random_leave_outs() -> None:
    meta = pd.DataFrame(
        {
            "network": ["Defaultmode"] * 4 + ["Negativesad"] * 4,
            "modality": ["connectivity"] * 4 + ["activation", "activation", "PPI", "PPI"],
            "contrast": ["task-free"] * 4 + ["AP_sad_vs_neutral", "FM_x_vs_y"] * 2,
            "tozzi_circuit": ["D_C", "D_C", "", "", "NS_A", "", "NS_C", ""],
        },
        index=[f"f{i}" for i in range(8)],
    )
    settings = {"random_sets": {"pools": ["all"], "leave_out_fractions": [0.5], "draws": 3}}
    sets = define_feature_sets(meta, settings, np.random.default_rng(4)).set_index("feature_set")
    assert sets.loc["tozzi_matched", "features"] == ["f0", "f1", "f4", "f6"]
    assert sets.loc["without_Negativesad", "features"] == ["f0", "f1", "f2", "f3"]
    assert sets.loc["without_AP_sad_vs_neutral", "n_features"] == 6
    assert "tozzi_task_free" not in sets.index  # two features: below the minimum of three
    random = sets.loc[sets.kind == "random"]
    assert len(random) == 3 and (random.n_features == 4).all()


def test_identical_feature_sets_agree_perfectly() -> None:
    values, _ = _clustered(np.random.default_rng(5), n=300)
    result = compare_clusterings(values, values.copy(), values, values, 6, **SETTINGS)
    assert result["ari"] == pytest.approx(1.0)
    assert result["mean_matched"] == pytest.approx(1.0)


def test_tozzi_profile_match_recovers_planted_biotypes() -> None:
    # distinct profiles: correlation distance cannot split near-collinear biotypes
    rng = np.random.default_rng(7)
    template = pd.DataFrame(rng.normal(0, 2, (6, 4)), columns=["D_C", "S_C", "A_C", "P_A"])
    template.index = [f"b{i}" for i in range(6)]
    circuits = np.repeat(template.columns.to_numpy(), 3)
    truth = rng.integers(0, 6, 900)
    matrix = template.to_numpy()[truth][:, np.arange(12) // 3] + rng.normal(0, 0.3, (900, 12))
    result = tozzi_profile_match(matrix, circuits, template, 6)
    assert result["n_profile_values"] == 4
    assert result["mean_matched"] > 0.95


def test_position_is_zero_at_the_null_and_one_at_the_planted_reference() -> None:
    rows = []
    for source, value in (("observed", 0.3), ("null", 0.2), ("planted", 0.6)):
        for split in range(3):
            rows.append(
                {
                    "feature_set": "s",
                    "source": source,
                    "split": split,
                    "mean_matched": value,
                    "weighted_matched": value,
                    "mean_unmatched": 0.0,
                    "separation": value,
                    "separation_centered": value,
                    "n_replicated": 1,
                    "smallest_cluster": 5,
                    "ari_assignment": value,
                }
            )
    summary = summarize_splits(pd.DataFrame(rows)).drop_duplicates("feature_set").iloc[0]
    assert summary.separation_position == pytest.approx(0.25)


def test_position_is_withheld_when_the_references_overlap() -> None:
    rng = np.random.default_rng(8)
    rows = [
        {
            "feature_set": "s",
            "source": source,
            "split": split,
            **{metric: rng.normal(0.5, 0.1) for metric in SUMMARY_METRICS},
        }
        for source in ("observed", "null", "planted")
        for split in range(40)
    ]
    summary = summarize_splits(pd.DataFrame(rows)).drop_duplicates("feature_set").iloc[0]
    assert 0.3 < summary.separation_auc < 0.7
    assert np.isnan(summary.separation_position)


def test_figure_tables_read_the_null_source_as_text(tmp_path) -> None:
    path = tmp_path / "table.tsv"
    pd.DataFrame({"source": ["observed", "null", "planted"], "pool": ["", "", "all"]}).to_csv(
        path, sep="\t", index=False
    )
    table = read_table(path)
    assert list(table.source) == ["observed", "null", "planted"]
    assert table.pool.isna().sum() == 2


def test_method_overview_renders(tmp_path) -> None:
    from for2107_repro.stability_overview import method_overview

    path = method_overview(tmp_path / "method.svg", n_patients=100, splits=10)
    assert path.stat().st_size > 10_000


def test_tozzi_codes_map_to_circuits() -> None:
    expected = {"D2D1": "D_C", "S1S2": "S_C", "A4A6": "A_C", "NS1": "NS_A", "NS2NS1": "NS_C",
                "NT3": "NTC_A", "NT2NT1": "NTC_C", "NTN1": "NTN_A", "NTN3NTN1": "NTN_C",
                "P2": "P_A", "C3": "C_A", "C1C2": "C_C"}  # fmt: skip
    assert {code: tozzi_code_circuit(code) for code in expected} == expected


def test_feature_keys_share_codes_across_measurements() -> None:
    planted = {
        "level": "feature",
        "feature_codes": {"D2D1_default_ampfc_ag_l_connectivity": "D2D1"},
    }
    names = [
        "D2D1_default_ampfc_ag_l_connectivity__AP",
        "D2D1_default_ampfc_ag_l_connectivity__REST",
        "x",
    ]
    assert list(planted_keys(names, ["D_C", "D_C", ""], planted)) == ["D2D1", "D2D1", ""]
    assert list(planted_keys(names, ["D_C", "D_C", ""], {"level": "circuit"})) == ["D_C", "D_C", ""]


def test_unavailable_tozzi_features_get_within_circuit_fill() -> None:
    observed = pd.DataFrame(
        [[1.0, 0.4, 0.1], [0.4, 1.0, 0.0], [0.1, 0.0, 1.0]],
        index=["NS1", "NS2", "D2D1"],
        columns=["NS1", "NS2", "D2D1"],
    )
    matrix = tozzi_correlation(observed, ["NS1", "NS2", "D2D1", "C1", "C2"])
    assert matrix[0, 1] == pytest.approx(0.4, abs=0.02)
    assert matrix[3, 4] == pytest.approx(0.4, abs=0.02)  # C_A filled with the task mean
    assert abs(matrix[0, 3]) < 0.02  # no correlation across circuits
    np.testing.assert_allclose(np.diag(matrix), 1.0)


def test_noise_covariance_keeps_the_total_when_given_as_total() -> None:
    from for2107_repro.replication import _mixture_moments

    correlation = 0.5 * np.ones((3, 3)) + 0.5 * np.eye(3)
    means = np.array([[0.4, -0.2, 0.1], [-0.3, 0.3, 0.0]])
    proportions = np.array([0.45, 0.55])
    within = noise_covariance(correlation, means, proportions, "within")
    total = noise_covariance(correlation, means, proportions, "total")
    np.testing.assert_allclose(within, correlation)
    np.testing.assert_allclose(_mixture_moments(means, proportions, total)[1], correlation)


def test_interpolated_scale_uses_the_first_crossing_of_the_target() -> None:
    # the p-value floors at 1 / (null draws + 1) beyond the crossing
    table = pd.DataFrame(
        {
            "scale": [0.8, 1.0, 1.1, 1.5, 2.0],
            "median_p_multinormal": [0.3, 0.134, 0.002, 0.002, 0.002],
        }
    )
    expected = 1.0 + 0.1 * np.log(0.134 / 0.016) / np.log(0.134 / 0.002)
    assert interpolate_scale(table, 0.016) == pytest.approx(expected)
    assert interpolate_scale(table.iloc[::-1], 0.016) == pytest.approx(expected)


def test_tozzi_split_half_r_is_high_with_and_without_subgroups() -> None:
    rng = np.random.default_rng(10)
    clustered, _ = _clustered(rng)
    assert np.nanmin(tozzi_split_half_r(clustered, 6, rng)) > 0.95
    r = tozzi_split_half_r(rng.normal(size=(400, 12)), 6, rng)
    assert r.shape == (6,) and np.nanmean(r) > 0.85


def test_permutation_test_rejects_correlated_data_without_subgroups() -> None:
    rng = np.random.default_rng(11)
    block = 0.6 * np.ones((4, 4)) + 0.4 * np.eye(4)
    factor = np.linalg.cholesky(np.kron(np.eye(3), block))
    silhouette, p = permutation_p(rng.standard_normal((400, 12)) @ factor.T, 6, 30, rng)
    assert p == 0.0 and silhouette > 0


def test_tozzi_checks_cover_every_source_and_repeat_exactly(tmp_path) -> None:
    rng = np.random.default_rng(12)
    template = pd.DataFrame({"D_C": [1.0, -1.0], "P_A": [0.5, 0.0]}, index=["b1", "b2"])
    references = build_references(
        rng.normal(size=(120, 8)),
        rng.normal(size=(120, 8)),
        np.array(["D_C", "P_A", "", "", "D_C", "", "", ""]),
        template,
        np.array([0.5, 0.5]),
        1.0,
    )
    settings = {"splits": 3, "permutation_datasets": 2, "permutations": 3, "n": 90}
    matrix = rng.normal(size=(120, 8))
    first = write_tozzi_checks(tmp_path, matrix, references, settings, 6, 7, 1)
    profile = read_table(tmp_path / "tozzi_split_half_r.tsv")
    permutation = read_table(tmp_path / "tozzi_permutation.tsv")
    assert profile.groupby("source").size().to_dict() == {"null": 3, "observed": 3, "planted": 3}
    assert permutation.groupby("source").n.agg(list).to_dict() == {
        "null": [90, 90],
        "observed": [120],
        "planted": [90, 90],
    }
    assert len(first["summary"]) == 6
    write_tozzi_checks(tmp_path, matrix, references, settings, 6, 7, 1)
    pd.testing.assert_frame_equal(read_table(tmp_path / "tozzi_permutation.tsv"), permutation)
    pd.testing.assert_frame_equal(read_table(tmp_path / "tozzi_split_half_r.tsv"), profile)


def _feature_dir(root: Path) -> None:
    """Exported feature folder: 30 controls, 28 MDD, 2 BD; sub-0030 moved in AP."""
    rng = np.random.default_rng(0)
    n = 60
    ids = [f"sub-{i:04d}" for i in range(n)]
    pd.DataFrame(
        {
            "subject_id": ids,
            "analysis_role": ["healthy_reference"] * 30 + ["clinical"] * 30,
            "diagnosis_group": ["HC"] * 30 + ["MDD"] * 28 + ["BD"] * 2,
            "site": np.tile(["Marburg", "Münster"], n // 2),
            "scanner_epoch": "epoch",
            "age": 30.0,
            "female": 1.0,
        }
    ).to_csv(root / "participants.tsv", sep="\t", index=False)
    motion = pd.DataFrame(
        {
            "subject_id": ids,
            "fd_threshold_mm": 0.3,
            "high_motion_pct_AP": 0.0,
            "n_frames_AP": 100,
            "high_motion_pct_FM": 0.0,
            "n_frames_FM": 100,
        }
    )
    # 40% of AP frames, 20% pooled over AP + FM
    motion.loc[30, "high_motion_pct_AP"] = 40.0
    motion.to_csv(root / "task_motion.tsv", sep="\t", index=False)
    rows = [
        (f"e{i}_connectivity{suffix}", task, "task-free", "connectivity", "")
        for i in (1, 2)
        for suffix, task in (("", "AP+FM"), ("__AP", "AP"), ("__FM", "FM"))
    ] + [
        ("ap_act", "AP", "sad > neutral", "activation", ""),
        ("ap_copy", "AP", "sad > neutral", "activation", "ap_act"),
        ("fm_act", "FM", "MatchFaces > MatchShapes", "activation", ""),
    ]
    catalog = pd.DataFrame(
        rows, columns=["feature", "source_task", "contrast", "modality", "duplicate_of"]
    ).assign(network="Net", tozzi_circuit="")
    catalog.to_csv(root / "task_feature_catalog.tsv", sep="\t", index=False)
    task = pd.DataFrame(rng.normal(size=(n, len(catalog))), columns=catalog.feature)
    task.insert(0, "subject_id", ids)
    task.to_csv(root / "task_features.tsv", sep="\t", index=False)
    rest = pd.DataFrame(rng.normal(size=(n, 2)), columns=["e1_connectivity", "e2_connectivity"])
    rest = rest.assign(e3_connectivity=rest.e1_connectivity, qc_pass=np.nan)
    rest.insert(0, "subject_id", ids)
    rest.to_csv(root / "rest_features.tsv", sep="\t", index=False)


def test_blocks_are_scored_from_the_feature_folder(tmp_path) -> None:
    _feature_dir(tmp_path)
    config = Config(
        Path("stability.yaml"),
        {
            "project": {"random_seed": 1},
            "stability": {
                "feature_dir": str(tmp_path),
                "clinical_groups": ["MDD"],
                "motion": {"max_fraction": 0.25},
            },
        },
    )
    blocks = prepare_blocks(config)["blocks"]
    assert set(blocks) == {"main", "edges_ap", "edges_fm", "edges_rest"}
    main = blocks["main"]
    # BD left out, duplicate column dropped
    assert len(main["scores"]) == 58 and main["clinical"].sum() == 28
    assert list(main["scores"]) == ["e1_connectivity", "e2_connectivity", "ap_act", "fm_act"]
    # the AP run is masked; the pooled task-free edges and the FM run are not
    assert main["missing"].loc["sub-0030", "ap_act"]
    assert not main["missing"].loc["sub-0030", ["e1_connectivity", "fm_act"]].any()
    assert "sub-0030" not in blocks["edges_ap"]["scores"].index
    assert not blocks["edges_fm"]["missing"].loc["sub-0030"].any()
    # the rest edge repeating another is dropped
    assert list(blocks["edges_rest"]["scores"]) == [
        "e1_connectivity__REST",
        "e2_connectivity__REST",
    ]


def test_incomplete_feature_folder_names_the_missing_table(tmp_path) -> None:
    _feature_dir(tmp_path)
    (tmp_path / "task_motion.tsv").unlink()
    config = Config(Path("stability.yaml"), {"stability": {"feature_dir": str(tmp_path)}})
    with pytest.raises(ConfigError, match="task_motion.tsv"):
        prepare_blocks(config)
