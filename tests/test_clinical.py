import numpy as np
import pandas as pd
from scipy import stats

from for2107_repro.clinical import characterize_clusters


def test_cluster_characterization_uses_independent_mann_whitney_test() -> None:
    data = pd.DataFrame(
        {
            "cluster": [1, 1, 1, 2, 2, 2, 2],
            "symptom": [8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    result = characterize_clusters(data, "cluster", ["symptom"])
    first = result.loc[result["cluster"] == 1].iloc[0]
    expected = stats.mannwhitneyu(
        [8.0, 9.0, 10.0],
        [1.0, 2.0, 3.0, 4.0],
        alternative="two-sided",
        method="asymptotic",
    )

    assert first["u"] == expected.statistic
    assert first["p_value"] == expected.pvalue
    assert np.isclose(first["effect_r"], 1.0)
