import numpy as np
import pandas as pd

from for2107_repro.harmonize import healthy_reference_scores, random_forest_pmm


def test_pmm_fills_missing_with_observed_donor() -> None:
    data = pd.DataFrame(
        {
            "scanner": ["a"] * 12,
            "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan, np.nan],
            "y": np.arange(12),
            "z": np.arange(12) ** 2,
        }
    )
    filled, audit = random_forest_pmm(data, ["x", "y", "z"], "scanner", seed=1)
    assert filled["x"].notna().all()
    assert set(filled.loc[10:, "x"]).issubset(set(data.loc[:9, "x"]))
    assert audit.loc[0, "method"] == "random-forest-pmm"


def test_parallel_pmm_is_numerically_identical() -> None:
    data = pd.DataFrame(
        {
            "scanner": ["a"] * 24,
            "x": [*range(20), np.nan, np.nan, np.nan, np.nan],
            "y": np.arange(24, dtype=float),
            "z": np.arange(24, dtype=float) ** 2,
        }
    )
    serial, serial_audit = random_forest_pmm(
        data, ["x", "y", "z"], "scanner", seed=31, n_jobs=1
    )
    parallel, parallel_audit = random_forest_pmm(
        data, ["x", "y", "z"], "scanner", seed=31, n_jobs=3
    )
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    pd.testing.assert_frame_equal(serial_audit, parallel_audit, check_exact=True)


def test_healthy_reference_scores() -> None:
    data = pd.DataFrame({"x": [0.0, 2.0, 4.0], "y": [2.0, 4.0, 8.0]})
    scores, audit = healthy_reference_scores(data, ["x", "y"], [True, True, False], minimum_n=2)
    assert np.isclose(scores.loc[:1, "x"].mean(), 0)
    assert np.isclose(scores.loc[:1, "x"].std(ddof=1), 1)
    assert list(audit["healthy_n"]) == [2, 2]
