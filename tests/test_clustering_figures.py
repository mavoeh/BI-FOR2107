import numpy as np
import pandas as pd

from for2107_repro.clustering import evaluate_solutions, fit_clusters
from for2107_repro.clustering_figures import supplementary_clustering_figures
from for2107_repro.tozzi_style import EDGE_SPEC


def test_supplementary_clustering_figure_set_is_complete() -> None:
    rng = np.random.default_rng(2107)
    matrix = rng.normal(size=(72, len(EDGE_SPEC)))
    labels = fit_clusters(matrix, 3).labels
    assignments = pd.DataFrame(
        {
            "subject_id": [f"sub-{index:04d}" for index in range(len(matrix))],
            "cluster": labels,
            **{
                f"{code}_test_connectivity": matrix[:, index]
                for index, code in enumerate(EDGE_SPEC)
            },
        }
    )
    evaluation = evaluate_solutions(matrix, [2, 3, 4])
    evaluation["p_multinormal"] = [0.2, 0.01, 0.04]
    evaluation["p_permuted"] = [0.3, 0.02, 0.06]
    loo = pd.DataFrame(
        {"n_clusters": [2, 3, 4], "loo_ari_mean": [0.7, 0.8, 0.75], "loo_ari_sd": [0.1] * 3}
    )
    subsample = pd.DataFrame(
        {
            "n_clusters": [2, 3, 4],
            "subsample_ari_mean": [0.5, 0.6, 0.55],
            "subsample_ari_sd": [0.1] * 3,
            "fraction": [0.8] * 3,
        }
    )

    figures = supplementary_clustering_figures(
        assignments,
        evaluation,
        loo,
        subsample,
        confirmatory=3,
        seed=2107,
    )

    assert set(figures) == {
        "supplementary_figure_1_elbow",
        "supplementary_figure_2_simulation_silhouette",
        "supplementary_figure_3_permutation_silhouette",
        "supplementary_figure_4_stability",
        "supplementary_figure_6_split_half_profiles",
    }
