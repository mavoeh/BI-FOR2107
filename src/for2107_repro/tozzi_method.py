"""Independent implementation of Tozzi-style task-residual and task features.

This module deliberately does not alter the existing external-rest pipeline.
The staged inputs are FSL FEAT preprocessing outputs, so task models are fitted
here to their filtered BOLD data and saved event tables.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
from scipy.ndimage import affine_transform
from scipy.signal import butter, sosfiltfilt

from .config import Config, ConfigError
from .features import load_feature_definitions

# Tozzi et al. (2024) feature definitions on the published ROI masks
# (cluster_study_2023, analysis_main/01_merge_data_qc.R). Task-free edges of
# these networks are listed in the feature map as published_edge. Every ROI
# of the task networks has an activation feature, and these pairs a PPI
# feature in both directions. Nonconscious threat uses an sgACC mask that is
# not among the published ROIs.
TOZZI_TASK_FREE_CIRCUITS = {"Defaultmode": "D", "Salience": "S", "Attention": "A"}
TOZZI_ACTIVATION_NETWORKS = ("Negativesad", "Negativethreat", "Positive", "Cognitive")
TOZZI_PPI_EDGES = frozenset(
    frozenset(edge)
    for edge in (
        ("Negativesad_AI_L", "Negativesad_pgACC_M"),
        ("Negativesad_AI_R", "Negativesad_pgACC_M"),
        ("Negativesad_Amy_L", "Negativesad_pgACC_M"),
        ("Negativesad_Amy_R", "Negativesad_pgACC_M"),
        ("Negativethreat_Amy_L", "Negativethreat_dACC_M"),
        ("Negativethreat_Amy_R", "Negativethreat_dACC_M"),
        ("Cognitive_dLPFCGFi_L", "Cognitive_dACC1_M"),
        ("Cognitive_dLPFCGFi1_R", "Cognitive_dACC1_M"),
    )
)


def roi_network(roi: str) -> str:
    """Tozzi ROI network of a mask name (its file prefix, e.g. ``Negativesad``)."""
    return roi.split("_", 1)[0]


@lru_cache(maxsize=40)
def _load_roi_image(path: str) -> nib.spatialimages.SpatialImage:
    """Cache the small immutable ROI masks across thousands of runs."""
    return nib.load(path)


def _fsl_voxel_coordinates(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    """FSL scaled-voxel coordinates for a NIfTI image (FLIRT convention)."""
    affine = image.affine
    zooms = image.header.get_zooms()[:3]
    matrix = np.diag([*zooms, 1.0])
    if np.linalg.det(affine[:3, :3]) > 0:
        matrix[0, 0] = -zooms[0]
        matrix[0, 3] = (image.shape[0] - 1) * zooms[0]
    return matrix


def _roi_in_functional_grid(
    roi_path: Path,
    functional_image: nib.spatialimages.SpatialImage,
    standard_image: nib.spatialimages.SpatialImage,
    func_to_standard: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour sample an MNI ROI into a FEAT functional grid.

    The FLIRT matrix maps functional scaled-voxel coordinates to standard
    scaled-voxel coordinates. This avoids treating a native-space FEAT series
    as if it were already in MNI space.
    """
    roi = _load_roi_image(str(roi_path))
    if roi.shape[:3] != standard_image.shape[:3] or not np.allclose(
        roi.affine, standard_image.affine, atol=1e-3
    ):
        raise ConfigError(
            f"ROI {roi_path.name} does not match FEAT standard reference {standard_image.shape[:3]}"
        )
    # output index is functional voxel; input coordinate is standard-ROI voxel
    map_std_to_func = (
        np.linalg.inv(_fsl_voxel_coordinates(roi))
        @ func_to_standard
        @ _fsl_voxel_coordinates(functional_image)
    )
    mask = (
        affine_transform(
            roi.get_fdata(dtype=np.float32) > 0,
            matrix=map_std_to_func[:3, :3],
            offset=map_std_to_func[:3, 3],
            output_shape=functional_image.shape[:3],
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        > 0
    )
    if not mask.any():
        raise ConfigError(f"ROI has no voxels in functional field of view: {roi_path}")
    return mask


def _continuous_map_in_functional_grid(
    source_path: Path,
    functional_image: nib.spatialimages.SpatialImage,
    source_reference: nib.spatialimages.SpatialImage,
    func_to_source: np.ndarray,
) -> np.ndarray:
    source = nib.load(str(source_path))
    if source.shape[:3] != source_reference.shape[:3] or not np.allclose(
        source.affine, source_reference.affine, atol=1e-3
    ):
        raise ConfigError(f"Nuisance tissue map is not aligned to its reference: {source_path}")
    source_to_func_voxel = (
        np.linalg.inv(_fsl_voxel_coordinates(source))
        @ func_to_source
        @ _fsl_voxel_coordinates(functional_image)
    )
    return affine_transform(
        np.asarray(source.dataobj, dtype=np.float32),
        matrix=source_to_func_voxel[:3, :3],
        offset=source_to_func_voxel[:3, 3],
        output_shape=functional_image.shape[:3],
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def _first_eigenvariate(voxels_by_time: np.ndarray) -> np.ndarray:
    """Return the first temporal principal component of ROI voxel signals."""
    data = np.asarray(voxels_by_time, dtype=np.float64)
    data = data[np.isfinite(data).all(axis=1)]
    if data.shape[0] == 0:
        raise ConfigError("ROI contains no finite voxels")
    data -= data.mean(axis=1, keepdims=True)
    scale = data.std(axis=1, ddof=1, keepdims=True)
    keep = scale[:, 0] > 0
    data = data[keep] / scale[keep]
    if not len(data):
        raise ConfigError("ROI contains no variable finite voxels")
    # SVD on voxel x time, with sign fixed for reproducibility.
    _, _, right = np.linalg.svd(data, full_matrices=False)
    component = right[0]
    if np.corrcoef(component, data.mean(axis=0))[0, 1] < 0:
        component = -component
    return component


def _design_matrix(
    events: pd.DataFrame, n_volumes: int, tr: float, motion: np.ndarray, tissue: np.ndarray
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    oversampling = 16
    dt = tr / oversampling
    n_high = n_volumes * oversampling
    conditions = sorted(events["trial_type"].astype(str).unique())
    regressors: dict[str, np.ndarray] = {}
    hrf = spm_hrf(dt, oversampling=1)
    for condition in conditions:
        neural = np.zeros(n_high, dtype=float)
        for row in events.loc[events.trial_type.astype(str) == condition].itertuples():
            first = max(0, int(np.floor(float(row.onset) / dt)))
            last = min(
                n_high, max(first + 1, int(np.ceil((float(row.onset) + float(row.duration)) / dt)))
            )
            neural[first:last] += 1
        convolved = np.convolve(neural, hrf, mode="full")[:n_high]
        regressors[condition] = convolved[::oversampling]
    names = list(regressors)
    task = np.column_stack([regressors[name] for name in names])
    # Remove intercept overlap and collinearity using an explicit intercept.
    nuisance = np.asarray(motion, dtype=float)
    if nuisance.shape != (n_volumes, 6) or tissue.shape != (n_volumes, 2):
        raise ConfigError(f"Expected six motion and two tissue confounds for {n_volumes} volumes")
    nuisance = nuisance - nuisance.mean(axis=0, keepdims=True)
    tissue = np.asarray(tissue, dtype=float)
    tissue = tissue - tissue.mean(axis=0, keepdims=True)
    matrix = np.column_stack([np.ones(n_volumes), task, nuisance, tissue])
    return matrix, names, regressors


def _fit_task_run(
    feat_dir: Path,
    events_path: Path,
    roi_names: list[str],
    roi_root: Path,
    task_name: str,
    config: Config,
) -> dict[str, Any]:
    bold_path = feat_dir / "filtered_func_data.nii.gz"
    standard_path = feat_dir / "reg" / "standard.nii.gz"
    transform_path = feat_dir / "reg" / "example_func2standard.mat"
    motion_path = feat_dir / "mc" / "prefiltered_func_data_mcf.par"
    anatomy_dir = (
        Path(config.require_path("inputs.task_data_root"))
        / "anat"
        / f"{feat_dir.name.removesuffix('.feat')}.anat"
    )
    highres_path = anatomy_dir / "T1_biascorr_brain.nii.gz"
    csf_path = anatomy_dir / "T1_fast_pve_0.nii.gz"
    wm_path = anatomy_dir / "T1_fast_pve_2.nii.gz"
    highres_transform_path = feat_dir / "reg" / "example_func2highres.mat"
    for required in (
        bold_path,
        standard_path,
        transform_path,
        motion_path,
        events_path,
        highres_path,
        csf_path,
        wm_path,
        highres_transform_path,
    ):
        if not required.exists():
            raise ConfigError(f"Missing task input: {required}")
    bold = nib.load(str(bold_path))
    if len(bold.shape) != 4:
        raise ConfigError(f"Task BOLD is not 4D: {bold_path}")
    data = np.asarray(bold.dataobj, dtype=np.float32)
    standard = nib.load(str(standard_path))
    tr_match = re.search(r"set fmri\(tr\)\s+([0-9.]+)", (feat_dir / "design.fsf").read_text())
    if not tr_match:
        raise ConfigError(f"Could not read TR in {feat_dir / 'design.fsf'}")
    tr = float(tr_match.group(1))
    motion = np.loadtxt(motion_path, dtype=float)
    if motion.ndim == 1:
        motion = motion.reshape(1, -1)
    if motion.shape[1] < 6:
        raise ConfigError(f"Motion file has fewer than six columns: {motion_path}")
    motion = motion[:, :6]
    if len(motion) != bold.shape[3]:
        raise ConfigError(f"Motion/BOLD length mismatch in {feat_dir}")
    highres = nib.load(str(highres_path))
    func_to_highres = np.loadtxt(highres_transform_path, dtype=float)
    if func_to_highres.shape != (4, 4):
        raise ConfigError(f"Expected 4x4 FLIRT transform in {highres_transform_path}")
    tissue_threshold = float(config.get("tozzi.tissue_probability_threshold", 0.9))
    tissue_signals = []
    for tissue_path in (csf_path, wm_path):
        probability = _continuous_map_in_functional_grid(
            tissue_path, bold, highres, func_to_highres
        )
        tissue_mask = probability >= tissue_threshold
        if tissue_mask.sum() < 5:
            raise ConfigError(f"Too few registered tissue voxels in {tissue_path}")
        tissue_signals.append(data[tissue_mask].mean(axis=0))
    tissue = np.column_stack(tissue_signals)
    events = pd.read_csv(events_path, sep="\t")
    if not {"onset", "duration", "trial_type"}.issubset(events):
        raise ConfigError(f"Events require onset, duration, trial_type: {events_path}")
    matrix, condition_names, condition_regressors = _design_matrix(
        events, bold.shape[3], tr, motion, tissue
    )
    transform = np.loadtxt(transform_path, dtype=float)
    if transform.shape != (4, 4):
        raise ConfigError(f"Expected 4x4 FLIRT transform in {transform_path}")

    # FEAT grand-mean scales each run and re-adds the temporal mean after
    # high-pass filtering, so a regional mean is still the local baseline.
    # Expressing activation/PPI betas relative to it removes scanner/coil
    # intensity differences. Residuals stay raw so that the concatenated
    # task-free connectivity features are identical across scalings.
    scaling = config.get("tozzi.signal_scaling", "raw")
    if scaling not in ("raw", "percent_signal_change"):
        raise ConfigError(f"Unknown tozzi.signal_scaling: {scaling}")

    # Load one image once; only the ROI voxel series are retained in memory.
    roi_series: dict[str, np.ndarray] = {}
    ppi_seed_series: dict[str, np.ndarray] = {}
    beta_scale: dict[str, float] = {}
    for roi_name in roi_names:
        mask = _roi_in_functional_grid(roi_root / f"{roi_name}.nii", bold, standard, transform)
        voxels = data[mask].reshape((-1, bold.shape[3]))
        finite = np.isfinite(voxels).all(axis=1)
        voxels = voxels[finite]
        if not len(voxels) or not np.isfinite(voxels).all():
            raise ConfigError(f"ROI contains no variable finite voxels: {roi_name}")
        if not np.any(voxels.std(axis=1) > 0):
            raise ConfigError(f"ROI contains no variable finite voxels: {roi_name}")
        # SPM estimates voxelwise GLMs and Tozzi averages the resulting
        # regional contrast maps. With a common design, fitting the mean raw
        # ROI signal gives the same mean beta and mean residual (unlike
        # voxelwise z-scoring, which changes the weighting).
        roi_series[roi_name] = voxels.mean(axis=0)
        ppi_seed_series[roi_name] = _first_eigenvariate(voxels)
        beta_scale[roi_name] = 1.0
        if scaling == "percent_signal_change":
            baseline = float(roi_series[roi_name].mean())
            if not baseline > 0:
                raise ConfigError(f"ROI has non-positive mean signal: {roi_name}")
            beta_scale[roi_name] = 100.0 / baseline

    # Residualize each ROI time series using the task, motion and WM/CSF
    # regressors. The 100-s FEAT high-pass filter has already been
    # applied upstream; do not apply a second high-pass here.
    series = np.column_stack(list(roi_series.values()))
    beta = np.linalg.lstsq(matrix, series, rcond=None)[0]
    residuals = series - matrix @ beta
    beta_by_roi = {name: beta[:, index].copy() for index, name in enumerate(roi_names)}
    residual_by_roi = {name: residuals[:, index].copy() for index, name in enumerate(roi_names)}

    contrasts = config.get("tozzi.task_contrasts", {})
    task_contrast = contrasts.get(task_name, {})
    activation: dict[str, float] = {}
    ppi: dict[str, float] = {}
    for label, spec in task_contrast.items():
        positive, negative = spec["positive"], spec["negative"]
        if positive not in condition_names or negative not in condition_names:
            continue
        for roi_name in spec.get("rois", roi_names):
            if roi_name not in beta_by_roi:
                continue
            activation[f"{task_name}_{label}_{roi_name}_activation_adapted"] = (
                float(
                    beta_by_roi[roi_name][1 + condition_names.index(positive)]
                    - beta_by_roi[roi_name][1 + condition_names.index(negative)]
                )
                * beta_scale[roi_name]
            )
        # Adapted psychophysiological interaction: target regression on the
        # task model, seed component and seed x psychological modulator. This
        # omits SPM's deconvolution/reconvolution and is explicitly labeled as
        # an adaptation, not an exact PPI replication.
        psychological = condition_regressors[positive] - condition_regressors[negative]
        for edge in spec.get("ppi_edges", config.get("tozzi.ppi_edges", [])):
            seed, target = edge
            if seed not in roi_series or target not in roi_series:
                continue
            name = f"{task_name}_{label}_{seed}__{target}_ppi_adapted"
            reciprocal = {}
            for physiological_roi, target_roi in ((seed, target), (target, seed)):
                physiological = ppi_seed_series[physiological_roi]
                interaction = physiological * psychological
                ppi_design = np.column_stack([matrix, physiological, psychological, interaction])
                ppi_beta = np.linalg.lstsq(ppi_design, roi_series[target_roi], rcond=None)[0]
                reciprocal[physiological_roi] = float(ppi_beta[-1]) * beta_scale[target_roi]
            # Tozzi standardizes each direction and averages afterwards; keep
            # both when configured so that order of operations can be matched.
            if config.get("tozzi.ppi_directions", "average") == "separate":
                for physiological_roi, value in reciprocal.items():
                    ppi[f"{name}__phys_{physiological_roi}"] = value
            else:
                ppi[name] = float(np.mean(list(reciprocal.values())))

    return {
        "roi_names": roi_names,
        "roi_series": roi_series,
        "residuals": residual_by_roi,
        "activation": activation,
        "ppi": ppi,
        "n_volumes": bold.shape[3],
        "tr": tr,
    }


def _bandpass(series: np.ndarray, tr: float, low: float, high: float) -> np.ndarray:
    nyquist = 0.5 / tr
    if not 0 < low < high < nyquist:
        raise ConfigError(f"Invalid bandpass {low}-{high} Hz for TR={tr}")
    sos = butter(2, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, np.asarray(series, dtype=float), axis=0)
    except ValueError as exc:
        raise ConfigError("Run is too short for zero-phase residual band-pass") from exc


def task_free_definitions(config: Config) -> list[Any]:
    return [
        item
        for item in load_feature_definitions(config)
        if item.source_task == "rest" and item.modality == "connectivity"
    ]


def extraction_rois(config: Config) -> list[str]:
    """ROIs extracted from every run: edge ROIs, then any further task ROIs."""
    roi_names = list(
        dict.fromkeys(name for item in task_free_definitions(config) for name in item.rois)
    )
    roi_names.extend(name for name in config.get("tozzi.task_rois", []) if name not in roi_names)
    return roi_names


def extract_tozzi_subject(
    config: Config, subject: str, ap_dir: Path, fm_dir: Path, ap_events: Path, fm_events: Path
) -> dict[str, Any]:
    definitions = task_free_definitions(config)
    roi_names = extraction_rois(config)
    roi_root = Path(config.require_path("features.roi_directory"))
    ap = _fit_task_run(ap_dir, ap_events, roi_names, roi_root, "AP", config)
    fm = _fit_task_run(fm_dir, fm_events, roi_names, roi_root, "FM", config)

    # Tozzi filters residuals separately within each run, then concatenates
    # runs before estimating task-free correlation features.
    low, high = (float(value) for value in config.get("tozzi.rest_band_hz", [0.009, 0.08]))
    filtered: dict[str, list[np.ndarray]] = {name: [] for name in roi_names}
    for run in (ap, fm):
        for name in roi_names:
            filtered[name].append(_bandpass(run["residuals"][name], run["tr"], low, high))
    combined = {name: np.concatenate(parts) for name, parts in filtered.items()}
    result: dict[str, Any] = {"subject_id": subject}
    for definition in definitions:
        left, right = definition.rois[:2]
        values = np.corrcoef(combined[left], combined[right])[0, 1]
        result[definition.name] = float(np.arctanh(np.clip(values, -0.999999, 0.999999)))
        # Optional single-run edges give a within-session reliability estimate.
        if config.get("tozzi.run_connectivity", False):
            for index, task in enumerate(("AP", "FM")):
                values = np.corrcoef(filtered[left][index], filtered[right][index])[0, 1]
                result[f"{definition.name}__{task}"] = float(
                    np.arctanh(np.clip(values, -0.999999, 0.999999))
                )
    for run in (ap, fm):
        result.update(run["activation"])
        result.update(run["ppi"])
    result["n_ap_volumes"] = ap["n_volumes"]
    result["n_fm_volumes"] = fm["n_volumes"]
    return result


def discover_task_pairs(config: Config) -> pd.DataFrame:
    """Find participants with both FEAT outputs and matching event tables."""
    root = config.require_path("inputs.task_data_root")
    session = str(config.get("selection.session", "ses-01"))
    rows = []
    task_dirs: dict[str, dict[tuple[str, str], Path]] = {"AP": {}, "FM": {}}
    for task in task_dirs:
        for feat in sorted((root / f"{task}_feat").glob("sub-*.feat")):
            match = re.match(r"(sub-[^_]+)_(ses-[^.]+)\.feat$", feat.name)
            if match:
                task_dirs[task][(match.group(1), match.group(2))] = feat
    for key in sorted(set(task_dirs["AP"]) | set(task_dirs["FM"])):
        subject, ses = key
        if session and ses != session:
            continue
        events = {}
        for task in ("AP", "FM"):
            if key not in task_dirs[task]:
                events[task] = None
                continue
            candidates = list(
                (root / "events" / task / subject / ses / "func").glob(
                    f"{subject}_{ses}_task-{task}*_events.tsv"
                )
            )
            events[task] = candidates[0] if len(candidates) == 1 else None
        reasons = []
        for task in ("AP", "FM"):
            if key not in task_dirs[task]:
                reasons.append(f"missing-{task}-feat")
            elif events[task] is None:
                reasons.append(f"missing-or-ambiguous-{task}-events")
        rows.append(
            {
                "subject_id": subject,
                "session": ses,
                "AP_feat": str(task_dirs["AP"][key]) if key in task_dirs["AP"] else None,
                "FM_feat": str(task_dirs["FM"][key]) if key in task_dirs["FM"] else None,
                "AP_events": str(events["AP"]) if events["AP"] else None,
                "FM_events": str(events["FM"]) if events["FM"] else None,
                "paired": key in task_dirs["AP"] and key in task_dirs["FM"],
                "ready": not reasons,
                "reason": ";".join(reasons) or "ok",
            }
        )
    return pd.DataFrame(rows)
