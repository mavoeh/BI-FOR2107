import numpy as np
import pandas as pd

from for2107_repro.report import (
    _bootstrap_median_interval,
    _mean_confidence_interval,
    _wilson_interval,
)
from for2107_repro.tozzi_style import (
    EDGE_SPEC,
    NODE_POSITIONS,
    cluster_deviation_tests,
    tozzi_style_rest_figure,
)


def test_mean_confidence_interval_contains_sample_mean() -> None:
    mean, half_width = _mean_confidence_interval(pd.Series([1.0, 2.0, 3.0, 4.0]))

    assert mean == 2.5
    assert half_width > 0


def test_wilson_interval_handles_zero_exceedances() -> None:
    low, high = _wilson_interval(0, 10_000)

    assert low == 0
    assert 0 < high < 0.001


def test_bootstrap_median_interval_is_deterministic() -> None:
    values = pd.Series(np.arange(1.0, 10.0))

    first = _bootstrap_median_interval(values, seed=2107, resamples=200)
    second = _bootstrap_median_interval(values, seed=2107, resamples=200)

    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_tozzi_style_rest_figure_contains_all_cluster_rows() -> None:
    rng = np.random.default_rng(2107)
    features = {f"{code}_test_connectivity": rng.normal(size=18) for code in EDGE_SPEC}
    assignments = pd.DataFrame(
        {
            "subject_id": [f"sub-{index:04d}" for index in range(18)],
            "cluster": np.repeat([1, 2, 3], 6),
            **features,
        }
    )

    figure = tozzi_style_rest_figure(assignments)

    assert len(figure.axes) == 6


def test_dmn_schematic_uses_published_anterior_posterior_geometry() -> None:
    assert NODE_POSITIONS["D1"][1] > 0
    assert NODE_POSITIONS["D2"][1] < NODE_POSITIONS["D4"][1] < 0
    assert NODE_POSITIONS["D3"][1] < NODE_POSITIONS["D4"][1] < 0


def test_cluster_deviation_tests_correct_jointly_across_edges() -> None:
    assignments = pd.DataFrame(
        {
            "cluster": np.repeat([1, 2], 20),
            "D2D1_test_connectivity": np.r_[
                np.linspace(0.8, 1.2, 20),
                np.tile([-1.0, 1.0], 10),
            ],
            "S1S3_test_connectivity": np.tile([-1.0, 1.0], 20),
        }
    )

    result = cluster_deviation_tests(assignments)

    strong = result.loc[
        (result["cluster"] == 1) & (result["feature"] == "D2D1_test_connectivity")
    ].iloc[0]
    assert strong["significant_fdr"]
    assert strong["ci_95_low"] > 0
    assert result["post_clustering_descriptive"].all()
