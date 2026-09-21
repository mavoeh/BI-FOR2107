import nibabel as nib
import numpy as np
import yaml

from for2107_repro.config import Config, load_config
from for2107_repro.features import (
    _roi_series,
    extract_rest_features,
    load_feature_definitions,
    within_circuit_connectivity,
)


def test_within_circuit_connectivity_is_mean_fisher_z() -> None:
    rng = np.random.default_rng(12)
    base = rng.normal(size=100)
    series = np.column_stack([base, base + rng.normal(scale=0.2, size=100), rng.normal(size=100)])
    corr = np.corrcoef(series, rowvar=False)
    expected = np.arctanh(np.clip(corr[np.triu_indices(3, 1)], -0.999999, 0.999999)).mean()
    assert np.isclose(within_circuit_connectivity(series), expected)


def test_roi_series_resamples_mask_to_bold_grid(tmp_path) -> None:
    bold_path = tmp_path / "bold.nii.gz"
    roi_path = tmp_path / "roi.nii.gz"
    bold = np.zeros((5, 5, 5, 8), dtype=np.float32)
    bold[2, 2, 2, :] = np.arange(8, dtype=np.float32)
    nib.save(nib.Nifti1Image(bold, np.eye(4)), bold_path)

    roi = np.zeros((10, 10, 10), dtype=np.uint8)
    roi[4:6, 4:6, 4:6] = 1
    nib.save(nib.Nifti1Image(roi, np.diag([0.5, 0.5, 0.5, 1.0])), roi_path)

    series = _roi_series(bold_path, [roi_path], None, 2.0, None, None)

    assert series.shape == (8, 1)
    assert np.isfinite(series).all()


def test_roi_series_fast_path_for_cleaned_same_grid_data(tmp_path) -> None:
    bold_path = tmp_path / "bold.nii.gz"
    roi_path = tmp_path / "roi.nii.gz"
    bold = np.zeros((3, 3, 3, 8), dtype=np.float32)
    bold[1, 1, 1, :] = np.arange(8, dtype=np.float32)
    bold[1, 1, 2, :] = np.arange(8, dtype=np.float32) ** 2
    nib.save(nib.Nifti1Image(bold, np.eye(4)), bold_path)
    roi = np.zeros((3, 3, 3), dtype=np.uint8)
    roi[1, 1, 1:3] = 1
    nib.save(nib.Nifti1Image(roi, np.eye(4)), roi_path)

    series = _roi_series(bold_path, [roi_path], None, 2.0, None, None, already_cleaned=True)

    expected = np.mean(
        [
            (bold[1, 1, 1] - bold[1, 1, 1].mean()) / bold[1, 1, 1].std(ddof=1),
            (bold[1, 1, 2] - bold[1, 1, 2].mean()) / bold[1, 1, 2].std(ddof=1),
        ],
        axis=0,
    )
    assert np.allclose(series[:, 0], expected)


def test_rest_circuits_share_one_extraction_without_changing_values(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(29)
    bold_path = tmp_path / "bold.nii.gz"
    bold = rng.normal(size=(4, 4, 4, 15)).astype(np.float32)
    nib.save(nib.Nifti1Image(bold, np.eye(4)), bold_path)

    roi_names = ["d1", "d2", "s1", "s2", "a1", "a2"]
    for index, name in enumerate(roi_names):
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask.reshape(-1)[index * 2 : index * 2 + 2] = 1
        nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / f"{name}.nii")

    mapping_path = tmp_path / "features.yaml"
    mapping_path.write_text(
        yaml.safe_dump(
            {
                "tiers": {"test": {"include_tasks": ["rest"]}},
                "features": [
                    {
                        "name": "default_mode_connectivity",
                        "circuit": "default_mode",
                        "source_task": "rest",
                        "modality": "connectivity",
                        "equivalence": "close",
                        "rois": ["d1", "d2"],
                        "ready": True,
                    },
                    {
                        "name": "salience_connectivity",
                        "circuit": "salience",
                        "source_task": "rest",
                        "modality": "connectivity",
                        "equivalence": "close",
                        "rois": ["s1", "s2"],
                        "ready": True,
                    },
                    {
                        "name": "attention_connectivity",
                        "circuit": "attention",
                        "source_task": "rest",
                        "modality": "connectivity",
                        "equivalence": "close",
                        "rois": ["a1", "a2"],
                        "ready": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = Config(
        tmp_path / "config.yaml",
        {
            "features": {
                "mapping": str(mapping_path),
                "tier": "test",
                "roi_directory": str(tmp_path),
                "input_is_cleaned": True,
                "confounds": [],
                "low_pass_hz": None,
                "high_pass_hz": None,
            },
            "preprocessing": {},
        },
    )
    definitions = load_feature_definitions(config)
    expected = {}
    for feature in definitions:
        series = _roi_series(
            bold_path,
            [tmp_path / f"{name}.nii" for name in feature.rois],
            None,
            1.0,
            None,
            None,
            already_cleaned=True,
        )
        expected[feature.name] = within_circuit_connectivity(series)

    from for2107_repro import features as feature_module

    original = feature_module._roi_series
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(feature_module, "_roi_series", counted)
    observed = extract_rest_features(config, "sub-test", bold_path, None, None, definitions)

    assert calls == 1
    for name, value in expected.items():
        assert observed[name] == value


def test_phase1_map_contains_the_fourteen_published_edges() -> None:
    definitions = load_feature_definitions(load_config("configs/external_rs_phase1.yaml"))

    assert len(definitions) == 14
    assert all(definition.source_task == "rest" for definition in definitions)
    assert all(definition.modality == "connectivity" for definition in definitions)
    assert all(definition.equivalence == "published_edge" for definition in definitions)
    assert all(len(definition.rois) == 2 for definition in definitions)
    assert [definition.name.split("_", 1)[0] for definition in definitions] == [
        "D2D1",
        "D1D3",
        "D1D4",
        "D2D4",
        "D3D4",
        "S1S3",
        "S2S4",
        "S1S2",
        "A2A1",
        "A3A1",
        "A4A2",
        "A5A3",
        "A4A6",
        "A5A7",
    ]
