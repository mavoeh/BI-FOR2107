"""Split-half reproducibility of six-cluster solutions across feature sets.

For each feature set the patients are split 50/50 at random, each half is
clustered with Tozzi's method (correlation distance, average linkage, k = 6)
and the cluster centroids of the two halves are compared by cosine
similarity after one-to-one (Hungarian) matching. Reproducible clusters give
a diagonal near 1 and low off-diagonal values.

A continuous distribution without clusters, cut into six groups along its
main axes, also reproduces across halves, and matching raises the diagonal
even for noise. Every set is therefore run with the same procedure on two
simulated references of the observed size: a multivariate normal with the
patients' covariance (no clusters), and Tozzi's six biotypes planted with his
per-feature biotype means (Supplementary Figure 5), scaled to the strength
of his evidence, on his features (within-biotype noise with the controls'
covariance). Cross-set comparisons
cluster the same patients with two feature sets; independent measurements
(AP run, FM run, AP + FM, rest scan) are compared on the same edges. Finally
the clusters are compared with Tozzi's circuit-level biotype profiles, and
his own split-half profile check and permutation test are run on FOR2107 and
both references.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from .clustering import fit_clusters
from .config import Config, ConfigError, load_config
from .feature_export import feature_files
from .progress import report, stage
from .provenance import write_provenance
from .replication import CIRCUIT_ORDER, faithful_scores, mask_high_motion
from .stability_figures import write_stability_figures

SOURCES = ("observed", "null", "planted")
SUMMARY_METRICS = (
    "mean_matched",
    "weighted_matched",
    "mean_unmatched",
    "separation",
    "separation_centered",
    "n_replicated",
    "smallest_cluster",
    "ari_assignment",
)
EDGE_SUFFIXES = {"edges_ap": "__AP", "edges_fm": "__FM", "edges_rest": "__REST"}


# --- similarity of cluster centroids --------------------------------------------


def cosine_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row of `first` with every row of `second`."""

    def unit(rows: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return np.divide(rows, norms, out=np.full_like(rows, np.nan), where=norms > 0)

    return unit(np.asarray(first, float)) @ unit(np.asarray(second, float)).T


def cluster_centroids(
    matrix: np.ndarray, labels: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Centroids and sizes of clusters 1..k; an empty cluster gets a NaN row."""
    sizes = np.array([np.sum(labels == cluster) for cluster in range(1, k + 1)])
    rows = [
        matrix[labels == cluster].mean(axis=0) if size else np.full(matrix.shape[1], np.nan)
        for cluster, size in zip(range(1, k + 1), sizes, strict=True)
    ]
    return np.vstack(rows), sizes


def matched_comparison(
    first: np.ndarray,
    first_sizes: np.ndarray,
    second: np.ndarray,
    second_sizes: np.ndarray,
    *,
    replicated: float,
    min_size: int,
) -> dict[str, Any]:
    """Match clusters one-to-one by cosine; rows ordered by size in `first`.

    In the returned k x k matrix row i is the i-th largest cluster of `first`
    and column i its match in `second`, so matched pairs lie on the diagonal.
    A cluster counts as replicated when its matched cosine reaches
    `replicated` and both clusters have at least `min_size` members.
    """
    k = len(first)
    similarity = cosine_matrix(first, second)
    rows, columns = linear_sum_assignment(-np.nan_to_num(similarity, nan=-2.0))
    match = dict(zip(rows, columns, strict=True))
    row_order = np.argsort(-first_sizes, kind="stable")
    column_order = np.array([match[row] for row in row_order])
    ordered = similarity[np.ix_(row_order, column_order)]
    diagonal = np.diag(ordered)
    sizes_first, sizes_second = first_sizes[row_order], second_sizes[column_order]
    present = (sizes_first > 0) & (sizes_second > 0)
    large = np.minimum(sizes_first, sizes_second) >= min_size
    mean_matched = float(np.nanmean(diagonal))
    mean_unmatched = float(np.nanmean(ordered[~np.eye(k, dtype=bool)]))
    return {
        "matrix": ordered,
        "matched": diagonal,
        "sizes_first": sizes_first,
        "mean_matched": mean_matched,
        "weighted_matched": float(
            np.average(diagonal[present], weights=(sizes_first + sizes_second)[present])
        ),
        "mean_unmatched": mean_unmatched,
        "separation": mean_matched - mean_unmatched,
        "n_replicated": int(np.sum(large & (diagonal >= replicated))),
        "smallest_cluster": int(min(sizes_first.min(), sizes_second.min())),
    }


def assign_to_centroids(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest centroid by profile correlation (Tozzi's split-half rule)."""

    def standardized(rows: np.ndarray) -> np.ndarray:
        centred = rows - rows.mean(axis=1, keepdims=True)
        return centred / np.linalg.norm(centred, axis=1, keepdims=True)

    valid = np.flatnonzero(np.isfinite(centroids).all(axis=1))
    correlation = standardized(values) @ standardized(centroids[valid]).T
    return valid[np.argmax(correlation, axis=1)] + 1


def split_half(
    matrix: np.ndarray, k: int, rng: np.random.Generator, *, replicated: float, min_size: int
) -> dict[str, Any]:
    """Cluster two random halves and compare their centroids."""
    order = rng.permutation(len(matrix))
    half = len(matrix) // 2
    first, second = matrix[order[:half]], matrix[order[half : 2 * half]]
    labels = [fit_clusters(part, k).labels for part in (first, second)]
    (c1, s1), (c2, s2) = (
        cluster_centroids(part, part_labels, k)
        for part, part_labels in zip((first, second), labels, strict=True)
    )
    result = matched_comparison(c1, s1, c2, s2, replicated=replicated, min_size=min_size)
    # without the half's mean profile: what distinguishes the clusters
    centred = matched_comparison(
        c1 - first.mean(axis=0),
        s1,
        c2 - second.mean(axis=0),
        s2,
        replicated=replicated,
        min_size=min_size,
    )
    result["separation_centered"] = centred["separation"]
    result["ari_assignment"] = float(
        adjusted_rand_score(labels[1], assign_to_centroids(second, c1))
    )
    return result


def compare_clusterings(
    first: np.ndarray,
    second: np.ndarray,
    profile_first: np.ndarray | None,
    profile_second: np.ndarray | None,
    k: int,
    *,
    replicated: float,
    min_size: int,
) -> dict[str, Any]:
    """Cluster the same participants twice; compare labels and shared profiles."""
    labels_first, labels_second = fit_clusters(first, k).labels, fit_clusters(second, k).labels
    result: dict[str, Any] = {"ari": float(adjusted_rand_score(labels_first, labels_second))}
    if profile_first is not None and profile_first.shape[1] >= 2:
        c1, s1 = cluster_centroids(profile_first, labels_first, k)
        c2, s2 = cluster_centroids(profile_second, labels_second, k)
        result.update(matched_comparison(c1, s1, c2, s2, replicated=replicated, min_size=min_size))
    return result


def tozzi_profile_match(
    matrix: np.ndarray, keys: np.ndarray, template: pd.DataFrame, k: int
) -> dict[str, Any]:
    """Cosine of cluster profiles with Tozzi's biotypes after matching.

    `keys` holds each column's entry in the template ("" for none): a Tozzi
    feature code, or a circuit, whose score is then the mean of its features
    as in Fig. 3.
    """
    labels = fit_clusters(matrix, k).labels
    available = [key for key in template.columns if np.any(keys == key)]
    centroids, sizes = cluster_centroids(matrix, labels, k)
    profiles = np.column_stack([centroids[:, keys == key].mean(axis=1) for key in available])
    similarity = cosine_matrix(profiles, template[available].to_numpy())
    rows, columns = linear_sum_assignment(-np.nan_to_num(similarity, nan=-2.0))
    matched = similarity[rows, columns]
    result: dict[str, Any] = {
        "n_profile_values": len(available),
        "mean_matched": float(np.nanmean(matched)),
    }
    for row, column, value in zip(rows, columns, matched, strict=True):
        result[f"cos_{template.index[column]}"] = float(value)
        result[f"n_{template.index[column]}"] = int(sizes[row])
    return result


# --- Tozzi's own checks ---------------------------------------------------------


def tozzi_split_half_r(matrix: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Tozzi's split-half check (Fig. 2f, Supplementary Figure 6).

    One random half is clustered, the other is assigned to its centroids by
    profile correlation, and each cluster's mean profile is correlated between
    the halves. Returns r per cluster, largest first-half cluster first (NaN
    where fewer than two are assigned). The assignment puts each patient with
    the profile it already resembles, so r is high with or without subgroups.
    """
    order = rng.permutation(len(matrix))
    half = len(matrix) // 2
    first, second = matrix[order[:half]], matrix[order[half:]]
    centroids, sizes = cluster_centroids(first, fit_clusters(first, k).labels, k)
    means, counts = cluster_centroids(second, assign_to_centroids(second, centroids), k)
    r = np.array(
        [np.corrcoef(centroids[i], means[i])[0, 1] if counts[i] > 1 else np.nan for i in range(k)]
    )
    return r[np.argsort(-sizes, kind="stable")]


def permutation_p(
    matrix: np.ndarray, k: int, permutations: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Mean silhouette at k and Tozzi's permutation p (perm_cluster_sig).

    Each feature is shuffled on its own; p is his plain share of shuffles
    with a higher silhouette. Shuffling removes the correlations between
    features, so correlated data beat it with or without subgroups.
    """
    from .heavy_tails import _silhouettes

    observed = float(_silhouettes(matrix, [k])[0])
    shuffled = np.array(
        [
            _silhouettes(np.column_stack([rng.permutation(column) for column in matrix.T]), [k])[0]
            for _ in range(permutations)
        ]
    )
    return observed, float(np.mean(shuffled > observed))


# --- simulated references -----------------------------------------------------


def covariance_factor(covariance: np.ndarray) -> np.ndarray:
    """Square root L of a covariance (L @ L.T), negative eigenvalues clipped.

    Rows can be subset: L[idx] @ L[idx].T is the covariance of columns idx.
    """
    values, vectors = np.linalg.eigh(np.asarray(covariance, float))
    return vectors * np.sqrt(np.clip(values, 0, None))


@dataclass(frozen=True)
class References:
    """Cluster-free and planted-biotype data with the observed dimensions."""

    null_mean: np.ndarray
    null_factor: np.ndarray
    planted_means: np.ndarray
    proportions: np.ndarray
    planted_factor: np.ndarray

    def subset(self, columns: np.ndarray) -> References:
        return References(
            self.null_mean[columns],
            self.null_factor[columns],
            self.planted_means[:, columns],
            self.proportions,
            self.planted_factor[columns],
        )

    def draw(self, source: str, n: int, rng: np.random.Generator) -> np.ndarray:
        if source == "null":
            noise = rng.standard_normal((n, self.null_factor.shape[1]))
            return self.null_mean + noise @ self.null_factor.T
        truth = rng.choice(len(self.proportions), size=n, p=self.proportions)
        noise = rng.standard_normal((n, self.planted_factor.shape[1]))
        return self.planted_means[truth] + noise @ self.planted_factor.T


def build_references(
    patients: np.ndarray,
    controls: np.ndarray,
    keys: np.ndarray,
    template: pd.DataFrame,
    proportions: np.ndarray,
    scale: float,
) -> References:
    """Null: the patients' mean and covariance. Planted: Tozzi's biotype means
    (scaled) on the columns with a template key (his feature code or circuit),
    zero elsewhere, with the controls' covariance as within-biotype noise."""
    means = np.array(
        [
            [scale * float(template.loc[biotype].get(key, 0.0)) if key else 0.0 for key in keys]
            for biotype in template.index
        ]
    )
    return References(
        patients.mean(axis=0),
        covariance_factor(np.cov(patients, rowvar=False)),
        means,
        proportions,
        covariance_factor(np.cov(controls, rowvar=False)),
    )


# --- process-pool tasks -------------------------------------------------------


def _split_half_task(task: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    name, matrix, references, k, splits, seed, settings = task
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    for source in SOURCES:
        stack = []
        for split in range(splits):
            data = matrix if source == "observed" else references.draw(source, len(matrix), rng)
            result = split_half(data, k, rng, **settings)
            stack.append(result.pop("matrix"))
            matched, sizes = result.pop("matched"), result.pop("sizes_first")
            rows.append(
                {
                    "feature_set": name,
                    "source": source,
                    "split": split,
                    **result,
                    **{f"matched_{rank + 1}": value for rank, value in enumerate(matched)},
                    **{f"size_{rank + 1}": int(value) for rank, value in enumerate(sizes)},
                }
            )
        matrices[source] = np.nanmean(np.stack(stack), axis=0)
    return rows, matrices


def _cross_task(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    """One comparison: observed once, then references drawn `draws` times each."""
    name, kind, data, columns, references, k, draws, seed, settings = task
    first, second, profile_first, profile_second = columns
    rng = np.random.default_rng(seed)
    rows = []
    for source in SOURCES:
        for draw in range(1 if source == "observed" else draws):
            values = data if source == "observed" else references.draw(source, len(data), rng)
            result = compare_clusterings(
                values[:, first],
                values[:, second],
                None if profile_first is None else values[:, profile_first],
                None if profile_second is None else values[:, profile_second],
                k,
                **settings,
            )
            for key in ("matrix", "matched", "sizes_first"):
                result.pop(key, None)
            rows.append(
                {"comparison": name, "kind": kind, "source": source, "draw": draw, **result}
            )
    return rows


def _tozzi_task(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    name, matrix, circuits, template, references, k, draws, seed = task
    rng = np.random.default_rng(seed)
    rows = []
    for source in SOURCES:
        for draw in range(1 if source == "observed" else draws):
            values = matrix if source == "observed" else references.draw(source, len(matrix), rng)
            result = tozzi_profile_match(values, circuits, template, k)
            rows.append({"feature_set": name, "source": source, "draw": draw, **result})
    return rows


def _profile_r_task(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    source, matrix, references, k, splits, seed = task
    rng = np.random.default_rng(seed)
    rows = []
    for split in range(splits):
        data = matrix if source == "observed" else references.draw(source, len(matrix), rng)
        r = tozzi_split_half_r(data, k, rng)
        rows.append(
            {
                "source": source,
                "split": split,
                "mean_r": float(np.nanmean(r)),
                **{f"r_{rank + 1}": float(value) for rank, value in enumerate(r)},
            }
        )
    return rows


def _permutation_task(task: tuple[Any, ...]) -> dict[str, Any]:
    source, dataset, matrix, references, k, n, permutations, seed = task
    rng = np.random.default_rng(seed)
    data = matrix if source == "observed" else references.draw(source, n, rng)
    silhouette, p = permutation_p(data, k, permutations, rng)
    return {
        "source": source,
        "dataset": dataset,
        "n": len(data),
        "silhouette": silhouette,
        "p_permuted": p,
    }


def _map(function: Any, tasks: list[Any], n_jobs: int) -> list[Any]:
    if n_jobs <= 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(function, tasks))


# --- feature sets ---------------------------------------------------------------


def define_feature_sets(
    meta: pd.DataFrame, settings: dict[str, Any], rng: np.random.Generator
) -> pd.DataFrame:
    """Structured leave-outs and random leave-out subsets of the feature pool.

    `meta` is indexed by feature with columns network, modality, contrast and
    tozzi_circuit ("" for features Tozzi did not define).
    """
    features = meta.index.to_numpy()
    tozzi = meta.tozzi_circuit != ""
    task_free_circuits = meta.tozzi_circuit.isin(["D_C", "S_C", "A_C"])
    structured: dict[str, pd.Series] = {
        "all": pd.Series(True, index=meta.index),
        "tozzi_matched": tozzi,
        "tozzi_task_free": task_free_circuits,
        "task_free_all": meta.modality == "connectivity",
        "extensions": ~tozzi,
    }
    for network in sorted(meta.network.unique()):
        structured[f"without_{network}"] = meta.network != network
    for modality in sorted(meta.modality.unique()):
        structured[f"without_{modality}"] = meta.modality != modality
    for contrast in sorted(set(meta.contrast) - {"task-free"}):
        structured[f"without_{contrast}"] = meta.contrast != contrast
    for circuit in sorted(set(meta.tozzi_circuit[tozzi])):
        structured[f"tozzi_without_{circuit}"] = tozzi & (meta.tozzi_circuit != circuit)
    rows = [
        {
            "feature_set": name,
            "kind": "structured",
            "pool": "",
            "leave_out": np.nan,
            "features": list(features[mask.to_numpy()]),
        }
        for name, mask in structured.items()
    ]
    random_settings = settings.get("random_sets", {}) or {}
    for pool in random_settings.get("pools", ["tozzi_matched", "all"]):
        candidates = features[structured[pool].to_numpy()]
        for fraction in random_settings.get("leave_out_fractions", [0.1, 0.25, 0.5, 0.75]):
            size = max(3, round(len(candidates) * (1 - float(fraction))))
            for draw in range(int(random_settings.get("draws", 10))):
                keep = set(rng.choice(candidates, size=size, replace=False))
                rows.append(
                    {
                        "feature_set": f"random_{pool}_drop{round(100 * float(fraction))}_{draw:02d}",
                        "kind": "random",
                        "pool": pool,
                        "leave_out": float(fraction),
                        "features": [name for name in candidates if name in keep],
                    }
                )
    table = pd.DataFrame(rows)
    table["n_features"] = table.features.map(len)
    return table.loc[table.n_features >= 3].reset_index(drop=True)


# --- data preparation -------------------------------------------------------------


def _feature_metadata(catalog: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Catalog row of each scored feature (PPI: of either direction)."""
    catalog = catalog.fillna("")
    by_base = catalog.assign(base=catalog.feature.str.split("__phys_").str[0]).drop_duplicates(
        "base"
    )
    meta = by_base.set_index("base").loc[features]
    meta["contrast"] = np.where(
        meta.contrast == "task-free",
        "task-free",
        meta.source_task + "_" + meta.contrast.str.replace(" > ", "_vs_"),
    )
    return meta[["network", "modality", "source_task", "contrast", "tozzi_circuit"]]


def prepare_blocks(config: Config) -> dict[str, Any]:
    """Score each measurement block separately (no imputation across blocks).

    Blocks: `main` (AP + FM task-free edges, activations and PPIs), and the
    edges from the AP run, the FM run and the rest scan alone. Inputs come
    from the exported feature folder (`stability.feature_dir`).
    """
    settings = config.get("stability", {}) or {}
    seed = int(config.get("project.random_seed", 2107))
    n_jobs = int(config.get("analysis.n_jobs", 1))
    files = feature_files(config.require_path("stability.feature_dir"))
    cohort = pd.read_csv(files["participants"], sep="\t")
    cohort = cohort.loc[
        (cohort.analysis_role == "healthy_reference")
        | cohort.diagnosis_group.isin(settings.get("clinical_groups", []))
    ].reset_index(drop=True)
    motion_settings = settings.get("motion", {}) or {}
    motion = pd.read_csv(files["motion"], sep="\t")
    catalog = pd.read_csv(files["task_catalog"], sep="\t").fillna("")
    catalog = catalog.loc[catalog.duplicate_of == ""]
    table = pd.read_csv(files["task_features"], sep="\t")
    connectivity = catalog.modality == "connectivity"
    groups = {
        "task_free": list(catalog.feature[connectivity & (catalog.source_task == "AP+FM")]),
        "AP": list(catalog.feature[~connectivity & (catalog.source_task == "AP")]),
        "FM": list(catalog.feature[~connectivity & (catalog.source_task == "FM")]),
        "edges_ap": list(catalog.feature[connectivity & (catalog.source_task == "AP")]),
        "edges_fm": list(catalog.feature[connectivity & (catalog.source_task == "FM")]),
    }
    motion_runs = {
        "task_free": ["AP", "FM"],
        "AP": ["AP"],
        "FM": ["FM"],
        "edges_ap": ["AP"],
        "edges_fm": ["FM"],
    }
    columns = [feature for group in groups.values() for feature in group]
    data = cohort.merge(table[["subject_id", *columns]], on="subject_id")
    rest = pd.read_csv(files["rest_features"], sep="\t")
    edges = [column for column in rest if column.endswith("_connectivity")]
    # drop columns identical to an earlier one (shared amygdala masks)
    unique = list(rest[edges].T.drop_duplicates().index)
    groups["edges_rest"] = [f"{edge}__REST" for edge in unique]
    rest = rest[["subject_id", *unique]].rename(
        columns=lambda c: f"{c}__REST" if c in unique else c
    )
    data = data.merge(rest, on="subject_id", how="left")
    data, _ = mask_high_motion(
        data,
        groups,
        motion_runs,
        motion,
        float(motion_settings.get("max_fraction", 0.25)),
    )
    block_columns = {
        "main": groups["task_free"] + groups["AP"] + groups["FM"],
        **{name: groups[name] for name in ("edges_ap", "edges_fm", "edges_rest")},
    }
    blocks: dict[str, dict[str, Any]] = {}
    for name, block in block_columns.items():
        rows = data[block].notna().any(axis=1)
        frame = data.loc[rows].reset_index(drop=True)
        healthy = frame.analysis_role == "healthy_reference"
        with stage(f"Score block {name} ({len(block)} columns, {int(rows.sum())} participants)"):
            scores, final, _ = faithful_scores(
                frame, block, str(settings.get("batch", "site")), healthy, seed=seed, n_jobs=n_jobs
            )
        missing = pd.DataFrame(
            {
                feature: frame[[c for c in block if c.split("__phys_")[0] == feature]]
                .isna()
                .any(axis=1)
                for feature in final
            }
        )
        scores.index = missing.index = frame.subject_id.to_numpy()
        blocks[name] = {
            "scores": scores[final],
            "missing": missing,
            "clinical": (frame.analysis_role == "clinical").to_numpy(),
        }
    meta = _feature_metadata(catalog, list(blocks["main"]["scores"].columns))
    edge_meta = _feature_metadata(catalog, groups["task_free"])
    return {"blocks": blocks, "meta": meta, "edge_meta": edge_meta}


def _template(planted: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    """Biotype x key means: per feature code (Supp. Fig. 5) or per circuit (Fig. 3)."""
    if planted.get("level", "circuit") == "feature":
        template = pd.read_csv(planted["feature_means"], sep="\t", index_col="biotype")
    else:
        template = pd.DataFrame(planted["biotype_means"]).T.reindex(columns=CIRCUIT_ORDER)
    sizes = planted["biotype_sizes"]
    proportions = np.array([sizes[name] for name in template.index], dtype=float)
    return template, proportions / proportions.sum()


def planted_keys(names: list[str], circuits: list[str], planted: dict[str, Any]) -> np.ndarray:
    """Template key of each feature: its Tozzi code or circuit ("" for none)."""
    if planted.get("level", "circuit") != "feature":
        return np.asarray(circuits, dtype=object)
    codes = planted["feature_codes"]
    return np.array([codes.get(_edge_base(name), "") for name in names], dtype=object)


def tozzi_code_circuit(code: str) -> str:
    """Circuit of a Tozzi feature code (Supp. Table 18): NS2NS1 -> NS_C, NT1 -> NTC_A."""
    parts = re.findall(r"([A-Z]+)\d", code)
    if not parts:
        raise ValueError(f"Not a Tozzi feature code: {code}")
    if parts[0] in ("D", "S", "A"):
        return f"{parts[0]}_C"
    circuit = {"NS": "NS", "NT": "NTC", "NTN": "NTN", "P": "P", "C": "C"}[parts[0]]
    return f"{circuit}_{'C' if len(parts) == 2 else 'A'}"


def tozzi_correlation(observed: pd.DataFrame, codes: list[str]) -> np.ndarray:
    """Correlation of all of Tozzi's features: FOR2107 control correlations where
    available; unavailable features get the mean within-circuit correlation of
    the available task circuits within their circuit and none elsewhere."""
    circuits = [tozzi_code_circuit(code) for code in codes]
    present = [code for code in codes if code in observed.index]
    within = [
        observed.loc[a, b]
        for i, a in enumerate(present)
        for b in present[i + 1 :]
        if tozzi_code_circuit(a) == tozzi_code_circuit(b)
        and tozzi_code_circuit(a) not in ("D_C", "S_C", "A_C")
    ]
    fill = float(np.mean(within)) if within else 0.0
    matrix = np.eye(len(codes))
    for i, a in enumerate(codes):
        for j, b in enumerate(codes):
            if i != j and a in observed.index and b in observed.index:
                matrix[i, j] = observed.loc[a, b]
            elif i != j and circuits[i] == circuits[j]:
                matrix[i, j] = fill
    values, vectors = np.linalg.eigh(matrix)
    matrix = vectors @ np.diag(np.clip(values, 1e-3, None)) @ vectors.T
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


def noise_covariance(
    correlation: np.ndarray, means: np.ndarray, proportions: np.ndarray, role: str
) -> np.ndarray:
    """Within-biotype noise: `correlation` itself ("within"), or what is left
    of it once the biotypes are carved out of it as the total ("total")."""
    if role == "within":
        return correlation
    if role == "total":
        from .heavy_tails import within_biotype_correlation

        return within_biotype_correlation(correlation, means, proportions)
    raise ConfigError(f"Unknown correlation role: {role}")


def for2107_tozzi_correlation(
    controls: pd.DataFrame,
    names: pd.Index,
    keys: np.ndarray,
    planted: dict[str, Any],
    template: pd.DataFrame,
) -> np.ndarray:
    """Correlation of Tozzi's features from the FOR2107 controls (tozzi_correlation)."""
    codes = {name: key for name, key in zip(names, keys, strict=True) if key}
    missing = set(planted["feature_codes"]) - set(codes)
    if missing:
        raise ConfigError(f"Mapped Tozzi features not in the feature pool: {sorted(missing)}")
    observed = controls[list(codes)].rename(columns=codes).corr()
    return tozzi_correlation(observed, list(template.columns))


def calibrate_planted_scale(
    template: pd.DataFrame,
    proportions: np.ndarray,
    correlation: np.ndarray,
    settings: dict[str, Any],
    seed: int,
    n_jobs: int,
    role: str = "within",
) -> tuple[float, pd.DataFrame]:
    """Scale at which Tozzi's own design reproduces his multinormal-null p.

    Biotype means are inflated by selection, so they are scaled until data
    planted with them (all his features, his n and k, unit-variance noise
    with these correlations) give his evidence, as in the power analysis.
    With role "total" the correlations are those of all patients and the
    noise is what the biotypes leave of them.
    """
    from .replication import _mixture_moments, power_scenario

    means = template.to_numpy(dtype=float)
    columns = np.arange(means.shape[1])
    rng = np.random.default_rng(seed)
    executor = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    rows = []
    try:
        for scale in settings.get("scales", [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4]):
            report(f"Planted-strength calibration: scale {scale}")
            noise = noise_covariance(correlation, means * float(scale), proportions, role)
            total = _mixture_moments(means * float(scale), proportions, noise)[1]
            result = power_scenario(
                means * float(scale),
                proportions,
                noise,
                columns,
                int(settings.get("n", 801)),
                int(settings.get("k", 6)),
                int(settings.get("replicates", 60)),
                int(settings.get("null_draws", 500)),
                rng,
                executor,
            )
            rows.append(
                {
                    "scale": float(scale),
                    **result,
                    # how far the patients' total correlation is from the given one
                    "total_max_abs_change": float(
                        np.abs(total - correlation).max() if role == "total" else np.nan
                    ),
                }
            )
    finally:
        if executor is not None:
            executor.shutdown()
    table = pd.DataFrame(rows)
    return interpolate_scale(table, float(settings.get("p_multinormal", 0.016))), table


def interpolate_scale(table: pd.DataFrame, target: float) -> float:
    """Smallest scale whose median p reaches `target`, interpolated on log p
    between the grid points around the first crossing. Beyond it the p-value
    sits at its floor, 1 / (null draws + 1), and says nothing more."""
    table = table.sort_values("scale")
    scales = table.scale.to_numpy(dtype=float)
    log_p = np.log(table.median_p_multinormal.clip(lower=1e-4).to_numpy(dtype=float))
    crossed = np.flatnonzero(log_p <= np.log(target))
    if not len(crossed) or crossed[0] == 0:
        report("Calibration target lies outside the scale grid; using the nearest end")
        return float(scales[-1] if not len(crossed) else scales[0])
    i = crossed[0]
    fraction = (log_p[i - 1] - np.log(target)) / (log_p[i - 1] - log_p[i])
    return float(scales[i - 1] + fraction * (scales[i] - scales[i - 1]))


def _edge_base(name: str) -> str:
    for suffix in EDGE_SUFFIXES.values():
        name = name.removesuffix(suffix)
    return name


# --- summaries ------------------------------------------------------------------


def _auc(higher: np.ndarray, lower: np.ndarray) -> float:
    """Probability that a value from `higher` exceeds one from `lower`."""
    higher, lower = higher[np.isfinite(higher)], lower[np.isfinite(lower)]
    if not len(higher) or not len(lower):
        return np.nan
    greater = (higher[:, None] > lower[None, :]).mean()
    ties = (higher[:, None] == lower[None, :]).mean()
    return float(greater + ties / 2)


def summarize_splits(splits: pd.DataFrame, min_auc: float = 0.75) -> pd.DataFrame:
    """Median and quartiles per feature set and source, plus each set's
    position between the null (0) and the planted reference (1).

    `auc` is how well a single split tells planted biotypes from the null; the
    position is reported only where it reaches `min_auc`, since it is
    meaningless when the two references overlap.
    """
    grouped = splits.groupby(["feature_set", "source"])[list(SUMMARY_METRICS)]
    summary = grouped.median().add_suffix("_median")
    summary = summary.join(grouped.quantile(0.25).add_suffix("_q25"))
    summary = summary.join(grouped.quantile(0.75).add_suffix("_q75")).reset_index()
    wide = summary.pivot(index="feature_set", columns="source")
    rows = []
    for name, group in splits.groupby("feature_set"):
        row: dict[str, Any] = {"feature_set": name}
        for metric in ("separation", "separation_centered", "ari_assignment", "mean_matched"):
            values = {source: wide.loc[name, (f"{metric}_median", source)] for source in SOURCES}
            row.update({f"{metric}_{source}": value for source, value in values.items()})
            auc = _auc(
                group.loc[group.source == "planted", metric].to_numpy(float),
                group.loc[group.source == "null", metric].to_numpy(float),
            )
            row[f"{metric}_auc"] = auc
            span = values["planted"] - values["null"]
            row[f"{metric}_position"] = (
                (values["observed"] - values["null"]) / span if auc >= min_auc else np.nan
            )
        rows.append(row)
    return summary.merge(pd.DataFrame(rows), on="feature_set")


def summarize_references(table: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Observed value against the null and planted draws (one-sided p)."""
    rows = []
    for key, group in table.groupby(keys):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        for metric in metrics:
            if metric not in group or group[metric].isna().all():
                continue
            observed = float(group.loc[group.source == "observed", metric].iloc[0])
            null = group.loc[group.source == "null", metric].dropna().to_numpy()
            planted = group.loc[group.source == "planted", metric].dropna().to_numpy()
            row[f"{metric}_observed"] = observed
            row[f"{metric}_null_median"] = float(np.median(null))
            row[f"{metric}_planted_median"] = float(np.median(planted))
            row[f"{metric}_p_null"] = float((np.sum(null >= observed) + 1) / (len(null) + 1))
            row[f"{metric}_planted_percentile"] = float(np.mean(planted <= observed))
        rows.append(row)
    return pd.DataFrame(rows)


def write_tozzi_checks(
    root: Path,
    matrix: np.ndarray,
    references: References,
    settings: dict[str, Any],
    k: int,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    """Tozzi's split-half profile r and permutation test on one feature set.

    The split-half check runs on random splits of the patients and on
    reference draws of their size; the permutation test on the patients and
    on reference datasets of his size (n = 801).
    """
    # a stream of its own, so the checks give the same numbers when run alone
    seeds = iter(np.random.SeedSequence([seed, 1]).generate_state(10_000))
    splits = int(settings.get("splits", 100))
    datasets = int(settings.get("permutation_datasets", 100))
    permutations = int(settings.get("permutations", 100))
    n = int(settings.get("n", 801))
    tasks = [(source, matrix, references, k, splits, next(seeds)) for source in SOURCES]
    profile = pd.DataFrame([row for rows in _map(_profile_r_task, tasks, n_jobs) for row in rows])
    tasks = [
        (source, dataset, matrix, references, k, n, permutations, next(seeds))
        for source in SOURCES
        for dataset in range(1 if source == "observed" else datasets)
    ]
    permutation = pd.DataFrame(_map(_permutation_task, tasks, n_jobs))
    profile.to_csv(root / "tozzi_split_half_r.tsv", sep="\t", index=False)
    permutation.to_csv(root / "tozzi_permutation.tsv", sep="\t", index=False)
    rows = []
    for check, table, column in (
        ("split_half_r", profile, "mean_r"),
        ("permutation_p", permutation, "p_permuted"),
    ):
        for source in SOURCES:
            values = table.loc[table.source == source, column]
            rows.append(
                {
                    "check": check,
                    "source": source,
                    "count": len(values),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "share_p_below_05": float((values < 0.05).mean())
                    if check == "permutation_p"
                    else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "tozzi_checks_summary.tsv", sep="\t", index=False)
    report("Tozzi's own checks:\n" + summary.round(3).to_string(index=False))
    published = settings.get("published_split_half_r")
    return {
        "feature_set": settings.get("feature_set", "tozzi_matched"),
        "published_split_half_r": float(np.mean(published)) if published else None,
        "summary": summary.to_dict(orient="records"),
    }


# --- orchestration ------------------------------------------------------------------


def run_stability(config: Config) -> dict[str, Any]:
    root = config.require_path("outputs.root")
    root.mkdir(parents=True, exist_ok=True)
    settings = config.get("stability", {}) or {}
    seed = int(config.get("project.random_seed", 2107))
    n_jobs = int(config.get("analysis.n_jobs", 1))
    k = int(settings.get("k", 6))
    splits = int(settings.get("splits", 100))
    draws = int(settings.get("reference_draws", 100))
    compare = {
        "replicated": float(settings.get("replicated_cosine", 0.8)),
        "min_size": int(settings.get("min_cluster_size", 10)),
    }
    planted = settings["planted"]
    template, proportions = _template(planted)
    seeds = iter(np.random.SeedSequence(seed).generate_state(10_000))

    with stage("Load cohort, motion and score feature blocks"):
        prepared = prepare_blocks(config)
    main, meta = prepared["blocks"]["main"], prepared["meta"]
    patients = main["scores"].loc[main["clinical"]]
    controls = main["scores"].loc[~main["clinical"]]
    report(f"Main pool: {len(patients)} patients, {len(controls)} controls, {len(meta)} features")
    meta.to_csv(root / "feature_metadata.tsv", sep="\t")

    sets = define_feature_sets(meta, settings, np.random.default_rng(next(seeds)))
    sets.assign(features=sets.features.map(";".join)).to_csv(
        root / "feature_sets.tsv", sep="\t", index=False
    )
    keys = planted_keys(list(meta.index), list(meta.tozzi_circuit), planted)
    if planted.get("level", "circuit") == "feature":
        with stage("Calibrate the strength of the planted biotypes"):
            scale, calibration = calibrate_planted_scale(
                template,
                proportions,
                for2107_tozzi_correlation(controls, meta.index, keys, planted, template),
                planted.get("calibration", {}) or {},
                next(seeds),
                n_jobs,
            )
        calibration.to_csv(root / "planted_calibration.tsv", sep="\t", index=False)
        report(f"Planted biotypes scaled by {scale:.3f}")
    else:
        scale = float(planted["scale"])
    references = build_references(
        patients.to_numpy(), controls.to_numpy(), keys, template, proportions, scale
    )
    position = {feature: index for index, feature in enumerate(meta.index)}
    matrix = patients.to_numpy()

    with stage(f"Split-half reproducibility of {len(sets)} feature sets ({splits} splits each)"):
        tasks = []
        for item in sets.itertuples():
            idx = np.array([position[name] for name in item.features])
            tasks.append(
                (
                    item.feature_set,
                    matrix[:, idx],
                    references.subset(idx),
                    k,
                    splits,
                    next(seeds),
                    compare,
                )
            )
        results = _map(_split_half_task, tasks, n_jobs)
    split_rows = [row for rows, _ in results for row in rows]
    matrices = dict(zip(sets.feature_set, (item for _, item in results), strict=True))
    split_table = pd.DataFrame(split_rows)
    split_table.to_csv(root / "split_half_splits.tsv", sep="\t", index=False)
    summary = summarize_splits(split_table).merge(
        sets[["feature_set", "kind", "pool", "leave_out", "n_features"]], on="feature_set"
    )
    summary.to_csv(root / "split_half_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "feature_set": name,
                "source": source,
                "row_rank": row + 1,
                "column_rank": column + 1,
                "cosine": value,
            }
            for name, by_source in matrices.items()
            for source, values in by_source.items()
            for (row, column), value in np.ndenumerate(values)
        ]
    ).to_csv(root / "mean_cosine_matrices.tsv", sep="\t", index=False)

    with stage("Cross-set comparisons on the same patients"):
        features_of = dict(zip(sets.feature_set, sets.features, strict=True))
        pairs = [tuple(pair) for pair in settings.get("cross_set_pairs", [])]
        unknown = {name for pair in pairs for name in pair} - set(features_of)
        if unknown:
            raise ConfigError(
                f"Unknown feature sets in stability.cross_set_pairs: {sorted(unknown)}"
            )
        random_names = list(sets.feature_set[sets.kind == "random"])
        pair_rng = np.random.default_rng(next(seeds))
        for _ in range(int(settings.get("random_cross_pairs", 0))):
            pairs.append(tuple(pair_rng.choice(random_names, size=2, replace=False)))
        tasks = []
        for first, second in pairs:
            a = np.array([position[name] for name in features_of[first]])
            b = np.array([position[name] for name in features_of[second]])
            common = np.intersect1d(a, b)
            profile = common if len(common) >= 2 else None
            union = np.union1d(a, b)
            local = {column: index for index, column in enumerate(union)}

            def remap(
                columns: np.ndarray | None, local: dict[int, int] = local
            ) -> np.ndarray | None:
                return None if columns is None else np.array([local[c] for c in columns])

            tasks.append(
                (
                    f"{first} | {second}",
                    "same_measurement",
                    matrix[:, union],
                    (remap(a), remap(b), remap(profile), remap(profile)),
                    references.subset(union),
                    k,
                    draws,
                    next(seeds),
                    compare,
                )
            )
        tasks.extend(
            _independent_tasks(
                prepared, settings, planted, template, proportions, scale, k, draws, seeds, compare
            )
        )
        cross = pd.DataFrame([row for rows in _map(_cross_task, tasks, n_jobs) for row in rows])
    cross.to_csv(root / "cross_set_draws.tsv", sep="\t", index=False)
    cross_summary = summarize_references(
        cross, ["comparison", "kind"], ["ari", "mean_matched", "separation", "n_replicated"]
    )
    cross_summary.to_csv(root / "cross_set_summary.tsv", sep="\t", index=False)

    with stage("Comparison with Tozzi's biotype profiles"):
        tasks = []
        for item in sets.itertuples():
            idx = np.array([position[name] for name in item.features])
            if len(set(keys[idx]) - {""}) < 3 or (
                item.kind == "random" and item.pool != "tozzi_matched"
            ):
                continue
            tasks.append(
                (
                    item.feature_set,
                    matrix[:, idx],
                    keys[idx],
                    template,
                    references.subset(idx),
                    k,
                    draws,
                    next(seeds),
                )
            )
        tozzi = pd.DataFrame([row for rows in _map(_tozzi_task, tasks, n_jobs) for row in rows])
    tozzi.to_csv(root / "tozzi_match_draws.tsv", sep="\t", index=False)
    tozzi_summary = summarize_references(
        tozzi, ["feature_set"], ["mean_matched", *[f"cos_{name}" for name in template.index]]
    )
    tozzi_summary.to_csv(root / "tozzi_match_summary.tsv", sep="\t", index=False)

    checks = None
    if settings.get("tozzi_checks"):
        with stage("Tozzi's own split-half and permutation checks"):
            check_settings = settings["tozzi_checks"]
            name = check_settings.get("feature_set", "tozzi_matched")
            idx = np.array([position[feature] for feature in features_of[name]])
            checks = write_tozzi_checks(
                root, matrix[:, idx], references.subset(idx), check_settings, k, seed, n_jobs
            )

    structured = list(sets.feature_set[sets.kind == "structured"])
    figures = write_stability_figures(root)
    tables = feature_files(config.require_path("stability.feature_dir")).values()
    write_provenance(root / "provenance.json", config.path, tables)
    status = {
        "mode": "split-half-stability",
        "patients": len(patients),
        "controls": len(controls),
        "features": len(meta),
        "feature_sets": len(sets),
        "splits": splits,
        "reference_draws": draws,
        "planted_level": planted.get("level", "circuit"),
        "planted_scale": scale,
        "tozzi_checks": checks,
        "figures": figures,
        "structured": summary.loc[summary.feature_set.isin(structured)]
        .drop_duplicates("feature_set")[
            [
                "feature_set",
                "n_features",
                "separation_observed",
                "separation_null",
                "separation_planted",
                "separation_auc",
                "separation_position",
            ]
        ]
        .to_dict(orient="records"),
    }
    (root / "status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    return status


def run_tozzi_checks(config: Config) -> dict[str, Any]:
    """Tozzi's own checks added to a finished stability run without repeating
    it: same feature set, planted scale and references (write_tozzi_checks)."""
    root = config.require_path("outputs.root")
    status_path = root / "status.json"
    if not status_path.exists():
        raise ConfigError(f"No stability run in {root}; run the stability command first")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    settings = config.get("stability", {}) or {}
    check_settings = settings.get("tozzi_checks", {}) or {}
    seed = int(config.get("project.random_seed", 2107))
    n_jobs = int(config.get("analysis.n_jobs", 1))
    planted = settings["planted"]
    template, proportions = _template(planted)
    with stage("Load cohort, motion and score feature blocks"):
        prepared = prepare_blocks(config)
    main, meta = prepared["blocks"]["main"], prepared["meta"]
    patients = main["scores"].loc[main["clinical"]]
    controls = main["scores"].loc[~main["clinical"]]
    keys = planted_keys(list(meta.index), list(meta.tozzi_circuit), planted)
    references = build_references(
        patients.to_numpy(),
        controls.to_numpy(),
        keys,
        template,
        proportions,
        float(status["planted_scale"]),
    )
    sets = pd.read_csv(root / "feature_sets.tsv", sep="\t", keep_default_na=False)
    name = check_settings.get("feature_set", "tozzi_matched")
    features = sets.set_index("feature_set").loc[name, "features"].split(";")
    idx = np.array([list(meta.index).index(feature) for feature in features])
    with stage("Tozzi's own split-half and permutation checks"):
        checks = write_tozzi_checks(
            root,
            patients.to_numpy()[:, idx],
            references.subset(idx),
            check_settings,
            int(settings.get("k", 6)),
            seed,
            n_jobs,
        )
    status["tozzi_checks"] = checks
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    return checks


def run_planted_robustness(config: Config) -> dict[str, Any]:
    """Planted strength under each correlation structure of Tozzi's features.

    The stability run calibrates with FOR2107 control correlations as the
    within-biotype noise. Here the same calibration is repeated with the
    correlations fitted to his null silhouette curve (calibrate-tozzi), once
    as the noise and once as the total of all his patients, on one grid and
    with the same random numbers.
    """
    root = config.require_path("outputs.root")
    root.mkdir(parents=True, exist_ok=True)
    settings = config.get("planted_robustness", {}) or {}
    seed = int(config.get("project.random_seed", 2107))
    n_jobs = int(config.get("analysis.n_jobs", 1))
    stability_config = load_config(settings["stability_config"])
    planted = stability_config.get("stability.planted")
    template, proportions = _template(planted)
    codes = list(template.columns)
    with stage("Load cohort, motion and score feature blocks"):
        prepared = prepare_blocks(stability_config)
    main, meta = prepared["blocks"]["main"], prepared["meta"]
    controls = main["scores"].loc[~main["clinical"]]
    keys = planted_keys(list(meta.index), list(meta.tozzi_circuit), planted)
    fitted = pd.read_csv(settings["correlation_file"], sep="\t", index_col=0).loc[codes, codes]
    structures = {
        "for2107_within": (
            for2107_tozzi_correlation(controls, meta.index, keys, planted, template),
            "within",
        ),
        "tozzi_within": (fitted.to_numpy(dtype=float), "within"),
        "tozzi_total": (fitted.to_numpy(dtype=float), "total"),
    }
    calibration = {
        **(planted.get("calibration", {}) or {}),
        **{key: settings[key] for key in ("scales", "replicates", "null_draws") if key in settings},
    }
    scales, tables = [], []
    for name, (correlation, role) in structures.items():
        with stage(f"Calibrate the planted strength with {name}"):
            scale, table = calibrate_planted_scale(
                template, proportions, correlation, calibration, seed, n_jobs, role=role
            )
        report(f"{name}: planted biotypes scaled by {scale:.3f}")
        scales.append({"structure": name, "role": role, "scale": scale})
        tables.append(table.assign(structure=name, role=role))
    pd.DataFrame(scales).to_csv(root / "planted_scales.tsv", sep="\t", index=False)
    pd.concat(tables).to_csv(root / "planted_calibration.tsv", sep="\t", index=False)
    status = {
        "mode": "planted-strength-robustness",
        "target_p_multinormal": float(calibration.get("p_multinormal", 0.016)),
        "scales": scales,
    }
    (root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def _independent_tasks(
    prepared: dict[str, Any],
    settings: dict[str, Any],
    planted: dict[str, Any],
    template: pd.DataFrame,
    proportions: np.ndarray,
    scale: float,
    k: int,
    draws: int,
    seeds: Any,
    compare: dict[str, Any],
) -> list[tuple[Any, ...]]:
    """Edges measured independently: AP run, FM run, AP + FM residuals, rest scan.

    Only patients without imputed values in both blocks enter; the reference
    covariance spans both blocks, so the null keeps the measured agreement of
    the edges but has no clusters.
    """
    blocks = prepared["blocks"]
    edge_meta = prepared["edge_meta"]
    main = blocks["main"]
    sources = {
        "edges_apfm": main,
        **{name: blocks[name] for name in EDGE_SUFFIXES if name in blocks},
    }
    tasks = []
    for first, second in settings.get("independent_pairs", []):
        if first not in sources or second not in sources:
            continue
        frames = []
        for name in (first, second):
            block = sources[name]
            scores = block["scores"]
            edges = [c for c in scores.columns if _edge_base(c) in edge_meta.index]
            frame = scores[edges].rename(columns=_edge_base)
            keep = block["clinical"] & ~block["missing"][edges].any(axis=1).to_numpy()
            control = ~block["clinical"] & ~block["missing"][edges].any(axis=1).to_numpy()
            frames.append((frame.loc[keep], frame.loc[control]))
        edges = [e for e in frames[0][0].columns if e in frames[1][0].columns]
        for subset_name, subset in (
            ("all_edges", edges),
            ("tozzi_edges", [e for e in edges if edge_meta.loc[e, "tozzi_circuit"] != ""]),
        ):
            patients_ids = frames[0][0].index.intersection(frames[1][0].index)
            control_ids = frames[0][1].index.intersection(frames[1][1].index)
            patients = np.hstack(
                [frames[i][0].loc[patients_ids, subset].to_numpy() for i in (0, 1)]
            )
            controls = np.hstack([frames[i][1].loc[control_ids, subset].to_numpy() for i in (0, 1)])
            keys = planted_keys(subset, list(edge_meta.loc[subset, "tozzi_circuit"]), planted)
            references = build_references(
                patients, controls, np.tile(keys, 2), template, proportions, scale
            )
            width = len(subset)
            a, b = np.arange(width), np.arange(width, 2 * width)
            report(f"Independent {first} vs {second} ({subset_name}): {len(patients_ids)} patients")
            tasks.append(
                (
                    f"{first} vs {second} ({subset_name})",
                    "independent_measurement",
                    patients,
                    (a, b, a, b),
                    references,
                    k,
                    draws,
                    next(seeds),
                    compare,
                )
            )
    return tasks
