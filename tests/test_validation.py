import numpy as np
import pandas as pd

from for2107_repro.validation import (
    evaluate_selection_criteria,
    leave_site_out_replication,
    nuisance_associations,
)


def test_solution_selection_requires_both_nulls_and_both_stability_gates() -> None:
    evaluation = pd.DataFrame(
        {
            "n_clusters": [2, 3],
            "silhouette": [0.4, 0.5],
            "p_multinormal": [0.01, 0.01],
            "p_permuted": [0.01, 0.20],
        }
    )
    loo = pd.DataFrame({"n_clusters": [2, 3], "loo_ari_mean": [0.8, 0.9], "loo_ari_sd": [0.1, 0.1]})
    subsample = pd.DataFrame(
        {
            "n_clusters": [2, 3],
            "subsample_ari_mean": [0.4, 0.5],
            "subsample_ari_sd": [0.1, 0.1],
            "iterations": [10, 10],
            "fraction": [0.8, 0.8],
        }
    )

    table, status = evaluate_selection_criteria(
        evaluation,
        loo,
        subsample,
        {
            "p_multinormal": 0.05,
            "p_permuted": 0.05,
            "loo_ari_mean": 0.75,
            "subsample_ari_mean": 0.28,
        },
        confirmatory_clusters=3,
    )

    assert table["passes_all_prespecified_criteria"].tolist() == [True, False]
    assert status["supported_candidates"] == [2]
    assert status["confirmatory_passes_all"] is False
    assert status["selection_is_automatic"] is False


def test_nuisance_checks_return_categorical_and_continuous_effects() -> None:
    data = pd.DataFrame(
        {
            "cluster": [1, 1, 1, 2, 2, 2],
            "site": ["A", "A", "B", "A", "B", "B"],
            "sex": ["f", "m", "f", "f", "m", "m"],
            "age": [20, 21, 22, 30, 31, 32],
        }
    )

    result = nuisance_associations(data, "cluster", categorical=["site", "sex"])

    assert set(result["variable"]) == {"site", "sex", "age"}
    assert result["p_fdr"].between(0, 1).all()


def test_leave_site_out_uses_site_specific_healthy_references() -> None:
    rng = np.random.default_rng(4)
    rows = []
    profile = np.array([2.0, 1.0, 0.0, -1.0, -2.0, 0.5])
    for site, offset in (("A", 5.0), ("B", -3.0)):
        for _index in range(12):
            rows.append(
                {
                    "site": site,
                    "analysis_role": "healthy_reference",
                    **{f"f{j}": offset + rng.normal(scale=0.2) for j in range(6)},
                }
            )
        for direction in (-1, 1):
            for _index in range(10):
                values = offset + direction * profile + rng.normal(scale=0.1, size=6)
                rows.append(
                    {
                        "site": site,
                        "analysis_role": "clinical",
                        **{f"f{j}": values[j] for j in range(6)},
                    }
                )
    data = pd.DataFrame(rows)

    result = leave_site_out_replication(
        data,
        [f"f{j}" for j in range(6)],
        "site",
        n_clusters=2,
        minimum_reference_n=10,
    )

    assert set(zip(result["training_site"], result["test_site"], strict=True)) == {
        ("A", "B"),
        ("B", "A"),
    }
    assert len(result) == 4
    assert (result["assignment_ari_vs_independent_test_clustering"] > 0.8).all()
    assert (result["profile_correlation"] > 0.8).all()
