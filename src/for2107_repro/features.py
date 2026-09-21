from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker

from .config import Config, ConfigError


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    circuit: str
    source_task: str
    modality: str
    equivalence: str
    rois: tuple[str, ...]
    contrast: str | None
    ready: bool


def load_feature_definitions(config: Config) -> list[FeatureDefinition]:
    path = Path(config.get("features.mapping"))
    with path.open(encoding="utf-8") as stream:
        mapping = yaml.safe_load(stream)
    tier = config.get("features.tier", "primary_multimodal")
    tiers = mapping.get("tiers", {})
    if tier not in tiers:
        raise ConfigError(f"Unknown feature tier {tier!r}")
    tasks = set(tiers[tier]["include_tasks"])
    return [
        FeatureDefinition(
            name=item["name"],
            circuit=item["circuit"],
            source_task=item["source_task"],
            modality=item["modality"],
            equivalence=item["equivalence"],
            rois=tuple(item.get("rois", [])),
            contrast=item.get("contrast"),
            ready=bool(item.get("ready", False)),
        )
        for item in mapping.get("features", [])
        if item["source_task"] in tasks
    ]


def validate_feature_definitions(config: Config) -> pd.DataFrame:
    roi_root = Path(config.get("features.roi_directory"))
    rows = []
    for feature in load_feature_definitions(config):
        missing_rois = [name for name in feature.rois if not (roi_root / f"{name}.nii").exists()]
        reasons = []
        if not feature.ready:
            reasons.append("mapping-not-final")
        if feature.modality == "activation" and not feature.contrast:
            reasons.append("contrast-missing")
        if not feature.rois:
            reasons.append("roi-list-empty")
        if missing_rois:
            reasons.append("roi-files-missing")
        rows.append(
            {
                **feature.__dict__,
                "rois": ",".join(feature.rois),
                "missing_rois": ",".join(missing_rois),
                "usable": not reasons,
                "validation": ",".join(reasons) or "ok",
            }
        )
    return pd.DataFrame(rows)


def _roi_series(
    bold_path: Path,
    roi_paths: list[Path],
    confounds: pd.DataFrame | None,
    repetition_time: float,
    low_pass: float | None,
    high_pass: float | None,
    already_cleaned: bool = False,
) -> np.ndarray:
    bold_image = nib.load(str(bold_path))
    roi_images = [nib.load(str(path)) for path in roi_paths]
    same_grid = all(
        image.shape == bold_image.shape[:3]
        and np.allclose(image.affine, bold_image.affine, atol=1e-5)
        for image in roi_images
    )
    if same_grid and already_cleaned and confounds is None:
        # External inputs and the published ROI masks share the exact 2-mm
        # MNI grid. Loading the compressed BOLD image once avoids decompressing
        # the ~280-MB file separately for every ROI. Subject-level parallelism
        # is capped by configuration to keep aggregate memory use predictable.
        data = np.asanyarray(bold_image.dataobj, dtype=np.float32)
        flat = data.reshape((-1, data.shape[-1]))
        time_series = []
        for image in roi_images:
            mask = np.asanyarray(image.dataobj).reshape(-1) > 0
            voxels = flat[mask]
            means = np.nanmean(voxels, axis=1, keepdims=True)
            scales = np.nanstd(voxels, axis=1, ddof=1, keepdims=True)
            valid = np.isfinite(scales[:, 0]) & (scales[:, 0] > 0)
            if not valid.any():
                raise ConfigError(f"ROI contains no variable finite voxels: {image.get_filename()}")
            standardized = (voxels[valid] - means[valid]) / scales[valid]
            time_series.append(np.nanmean(standardized, axis=0))
        return np.column_stack(time_series)

    columns = None if confounds is None else confounds.replace([np.inf, -np.inf], np.nan).fillna(0)
    time_series = []
    for roi in roi_paths:
        # Nilearn 0.13 removed NiftiMasker's ``resampling_target`` argument.
        # Resample each binary ROI explicitly to the fMRIPrep BOLD grid so the
        # functional data are never interpolated onto the atlas grid.
        roi_in_bold_space = resample_to_img(
            str(roi),
            str(bold_path),
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        masker = NiftiMasker(
            mask_img=roi_in_bold_space,
            standardize="zscore_sample",
            detrend=not already_cleaned,
            low_pass=None if already_cleaned else low_pass,
            high_pass=None if already_cleaned else high_pass,
            t_r=repetition_time,
        )
        voxels = masker.fit_transform(str(bold_path), confounds=columns)
        time_series.append(np.nanmean(voxels, axis=1))
    return np.column_stack(time_series)


def within_circuit_connectivity(time_series: np.ndarray) -> float:
    if time_series.ndim != 2 or time_series.shape[1] < 2:
        raise ValueError("At least two ROI time series are required")
    correlations = np.corrcoef(time_series, rowvar=False)
    values = correlations[np.triu_indices_from(correlations, k=1)]
    values = np.clip(values, -0.999999, 0.999999)
    return float(np.nanmean(np.arctanh(values)))


def extract_rest_features(
    config: Config,
    subject: str,
    bold_path: Path,
    confounds_path: Path | None,
    metadata_path: Path | None,
    definitions: list[FeatureDefinition] | None = None,
) -> dict[str, Any]:
    if metadata_path:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        repetition_time = float(metadata["RepetitionTime"])
    else:
        image = nib.load(str(bold_path))
        if len(image.shape) != 4:
            raise ConfigError(f"Resting-state image is not 4D: {bold_path}")
        repetition_time = float(image.header.get_zooms()[3])
    confounds_table = (
        pd.read_csv(confounds_path, sep="\t") if confounds_path else pd.DataFrame()
    )
    requested_confounds = config.get("features.confounds", [])
    available = [column for column in requested_confounds if column in confounds_table]
    confounds = confounds_table[available] if available else None
    roi_root = Path(config.get("features.roi_directory"))
    result: dict[str, Any] = {"subject_id": subject}
    definitions = definitions if definitions is not None else load_feature_definitions(config)
    rest_features = [
        feature
        for feature in definitions
        if feature.source_task == "rest" and feature.modality == "connectivity"
    ]
    # Extract every unique ROI in one pass. For compressed external scans this
    # changes three full NIfTI decompressions (one per circuit) into one while
    # retaining the exact ROI order and calculations used previously.
    roi_names = list(dict.fromkeys(name for feature in rest_features for name in feature.rois))
    roi_paths = [roi_root / f"{name}.nii" for name in roi_names]
    missing = [str(path) for path in roi_paths if not path.exists()]
    if missing:
        raise ConfigError(f"Missing ROI masks: {missing}")
    if rest_features:
        all_series = _roi_series(
            bold_path,
            roi_paths,
            confounds,
            repetition_time,
            config.get("features.low_pass_hz"),
            config.get("features.high_pass_hz"),
            bool(config.get("features.input_is_cleaned", False)),
        )
        roi_columns = {name: index for index, name in enumerate(roi_names)}
        for feature in rest_features:
            columns = [roi_columns[name] for name in feature.rois]
            result[feature.name] = within_circuit_connectivity(all_series[:, columns])
    if "framewise_displacement" in confounds_table:
        fd = confounds_table["framewise_displacement"].fillna(0)
        threshold = float(config.get("preprocessing.fd_threshold_mm", 0.5))
        result["high_motion_fraction"] = float((fd > threshold).mean())
        result["qc_pass"] = result["high_motion_fraction"] <= float(
            config.get("preprocessing.max_high_motion_fraction", 0.25)
        )
        result["motion_qc_available"] = True
    else:
        result["high_motion_fraction"] = np.nan
        result["qc_pass"] = np.nan
        result["motion_qc_available"] = False
    return result


def extract_activation_feature(
    bold_path: Path,
    events_path: Path,
    confounds_path: Path,
    metadata_path: Path,
    roi_paths: list[Path],
    contrast: str,
) -> float:
    """Estimate a configured task contrast and average its effect across ROIs."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    events = pd.read_csv(events_path, sep="\t")
    required = {"onset", "duration", "trial_type"}
    if missing := required.difference(events):
        raise ConfigError(f"Events file lacks columns: {sorted(missing)}")
    confounds = pd.read_csv(confounds_path, sep="\t").replace([np.inf, -np.inf], np.nan).fillna(0)
    model = FirstLevelModel(
        t_r=float(metadata["RepetitionTime"]),
        noise_model="ar1",
        standardize=False,
        signal_scaling=0,
        drift_model="cosine",
        high_pass=0.008,
        minimize_memory=True,
    ).fit(str(bold_path), events=events, confounds=confounds)
    effect = model.compute_contrast(contrast, output_type="effect_size")
    values = []
    for roi_path in roi_paths:
        roi_in_effect_space = resample_to_img(
            str(roi_path),
            effect,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        masker = NiftiMasker(mask_img=roi_in_effect_space)
        values.append(float(np.nanmean(masker.fit_transform(effect))))
    return float(np.nanmean(values))
