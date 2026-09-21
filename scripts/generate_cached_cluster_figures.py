from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from for2107_repro.clustering import empirical_p_value, fit_clusters
from for2107_repro.clustering_figures import supplementary_clustering_figures
from for2107_repro.tozzi_style import cluster_deviation_tests, tozzi_style_rest_figure


def _correct_legacy_p_values(evaluation: pd.DataFrame) -> pd.DataFrame:
    result = evaluation.copy()
    for p_column, trial_column in (
        ("p_permuted", "permutations"),
        ("p_multinormal", "simulations"),
    ):
        counts = np.rint(result[p_column] * result[trial_column]).astype(int)
        result[p_column] = [
            empirical_p_value(int(count), int(trials))
            for count, trials in zip(counts, result[trial_column], strict=True)
        ]
    return result


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.svg", format="svg", bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    figure.savefig(output / f"{stem}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cluster figures from cached normalized features and validation tables."
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2107)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(args.analysis_dir / "cluster_assignments.tsv", sep="\t")
    features = [column for column in assignments if column not in {"subject_id", "cluster"}]
    fit = fit_clusters(assignments[features], args.clusters)
    assignments["cluster"] = fit.labels
    assignments.to_csv(
        args.output_dir / f"cluster_assignments_k{args.clusters}.tsv", sep="\t", index=False
    )

    evaluation = _correct_legacy_p_values(
        pd.read_csv(args.analysis_dir / "solution_evaluation.tsv", sep="\t")
    )
    loo = pd.read_csv(args.analysis_dir / "loo_stability.tsv", sep="\t")
    subsample = pd.read_csv(args.analysis_dir / "subsample_stability.tsv", sep="\t")
    figures = supplementary_clustering_figures(
        assignments,
        evaluation,
        loo,
        subsample,
        confirmatory=args.clusters,
        seed=args.seed,
    )
    captions: dict[str, str] = {}
    for stem, (figure, caption, artifacts) in figures.items():
        _save_figure(figure, args.output_dir, stem)
        captions[stem] = caption
        for artifact_name, artifact in artifacts.items():
            if isinstance(artifact, dict):
                for panel_name, table in artifact.items():
                    safe_panel = panel_name.lower().replace(" ", "_")
                    table.to_csv(
                        args.output_dir / f"{stem}_{artifact_name}_{safe_panel}.tsv",
                        sep="\t",
                        index=False,
                    )

    deviations = cluster_deviation_tests(assignments, features)
    deviations.to_csv(
        args.output_dir / f"tozzi_style_resting_profiles_k{args.clusters}_statistics.tsv",
        sep="\t",
        index=False,
    )
    profile_stem = f"tozzi_style_resting_profiles_k{args.clusters}"
    _save_figure(
        tozzi_style_rest_figure(assignments, deviation_tests=deviations),
        args.output_dir,
        profile_stem,
    )
    captions[profile_stem] = (
        "Five-cluster resting-state profiles; bars show means and unadjusted 95% confidence "
        "intervals, and circuit colors require BH-FDR significance across all cluster-edge tests."
    )
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(
            {
                "clusters": args.clusters,
                "source": str(args.analysis_dir),
                "feature_count": len(features),
                "participant_count": len(assignments),
                "cluster_sizes": {
                    str(key): int(value)
                    for key, value in assignments["cluster"].value_counts().sort_index().items()
                },
                "captions": captions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
