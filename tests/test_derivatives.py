from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from for2107_repro.config import Config, load_config
from for2107_repro.derivatives import discover_preprocessed_runs, write_available_features


def test_external_clean_rest_discovery_without_sidecars(tmp_path) -> None:
    bold = tmp_path / "sub-0042_ses-01_filtered_func_data_clean_MNI.nii.gz"
    bold.touch()
    original = load_config("configs/external_rs.yaml")
    data = {
        **original.data,
        "inputs": {
            **original.section("inputs"),
            "preprocessed_rest_root": str(tmp_path),
        },
    }
    runs = discover_preprocessed_runs(Config(Path("external.yaml"), data))

    assert runs[["subject_id", "session", "task"]].to_dict("records") == [
        {"subject_id": "sub-0042", "session": "ses-01", "task": "rest"}
    ]
    assert runs.loc[0, "confounds"] is None
    assert runs.loc[0, "metadata"] is None
    assert bool(runs.loc[0, "ready"])


def test_invalid_rest_roi_excludes_subject_and_continues(tmp_path) -> None:
    bold_path = tmp_path / "sub-0042_ses-01_filtered_func_data_clean_MNI.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((3, 3, 3, 8), dtype=np.float32), np.eye(4)),
        bold_path,
    )
    for name, location in (("roi_a", 1), ("roi_b", 2)):
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        mask.reshape(-1)[location] = 1
        nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / f"{name}.nii")
    mapping_path = tmp_path / "features.yaml"
    mapping_path.write_text(
        yaml.safe_dump(
            {
                "tiers": {"test": {"include_tasks": ["rest"]}},
                "features": [
                    {
                        "name": "rest_connectivity",
                        "circuit": "rest",
                        "source_task": "rest",
                        "modality": "connectivity",
                        "equivalence": "test",
                        "rois": ["roi_a", "roi_b"],
                        "ready": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = Config(
        tmp_path / "config.yaml",
        {
            "inputs": {
                "preprocessed_rest_root": str(tmp_path),
                "preprocessed_rest_pattern": "sub-*_ses-*_filtered_func_data_clean_MNI.nii.gz",
            },
            "outputs": {"root": str(tmp_path / "outputs")},
            "selection": {"session": "ses-01"},
            "features": {
                "mapping": str(mapping_path),
                "tier": "test",
                "roi_directory": str(tmp_path),
                "input_is_cleaned": True,
                "confounds": [],
                "low_pass_hz": None,
                "high_pass_hz": None,
                "extraction_jobs": 1,
            },
            "preprocessing": {
                "fd_threshold_mm": 0.5,
                "max_high_motion_fraction": 0.25,
            },
        },
    )

    result = write_available_features(config)

    assert result["features"].empty
    assert result["status"]["participants"] == 0
    assert result["status"]["skipped_participants"] == 1
    assert result["audit"].loc[0, "subject_id"] == "sub-0042"
    assert (
        result["audit"].loc[0, "reason"]
        == "subject-excluded-rest-feature-extraction-failed"
    )
    assert "ROI contains no variable finite voxels" in result["audit"].loc[0, "detail"]
