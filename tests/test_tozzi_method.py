from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from for2107_repro.config import Config
from for2107_repro.tozzi_method import (
    _bandpass,
    _design_matrix,
    _fit_task_run,
    _roi_in_functional_grid,
    discover_task_pairs,
)


def test_design_matrix_has_task_motion_and_tissue_regressors() -> None:
    events = pd.DataFrame({"onset": [2.0, 12.0], "duration": [0.1, 0.1], "trial_type": ["a", "b"]})
    design, names, regressors = _design_matrix(
        events, 40, 2.0, np.zeros((40, 6)), np.zeros((40, 2))
    )
    assert names == ["a", "b"]
    assert design.shape == (40, 11)  # intercept + 2 conditions + 6 motion + 2 tissue
    assert set(regressors) == {"a", "b"}
    assert np.isfinite(design).all()


def test_bandpass_preserves_inband_and_attenuates_outband() -> None:
    tr = 2.0
    time = np.arange(1000) * tr
    signal = np.sin(2 * np.pi * 0.04 * time) + np.sin(2 * np.pi * 0.15 * time)
    filtered = _bandpass(signal, tr, 0.009, 0.08)
    interior = slice(100, -100)
    expected = np.sin(2 * np.pi * 0.04 * time[interior])
    assert np.corrcoef(filtered[interior], expected)[0, 1] > 0.98


def test_roi_transform_identity_keeps_same_grid_mask(tmp_path: Path) -> None:
    data = np.zeros((8, 8, 8), dtype=np.uint8)
    data[2:5, 2:5, 2:5] = 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    roi_path = tmp_path / "roi.nii"
    nib.save(nib.Nifti1Image(data, affine), roi_path)
    image = nib.load(str(roi_path))
    result = _roi_in_functional_grid(roi_path, image, image, np.eye(4))
    assert np.array_equal(result, data.astype(bool))


def test_discovery_requires_both_task_runs_and_events(tmp_path: Path) -> None:
    from for2107_repro.config import load_config

    # Use the repository config but point it at a tiny synthetic folder tree.
    root = tmp_path / "tasks"
    for task in ("AP", "FM"):
        feat = root / f"{task}_feat" / "sub-0001_ses-01.feat"
        feat.mkdir(parents=True)
        event_dir = root / "events" / task / "sub-0001" / "ses-01" / "func"
        event_dir.mkdir(parents=True)
        (event_dir / f"sub-0001_ses-01_task-{task}_events.tsv").touch()
    config_path = tmp_path / "config.yaml"
    repo_config = load_config("configs/tozzi_for2107.yaml")
    import yaml

    raw = dict(repo_config.data)
    raw["inputs"]["task_data_root"] = str(root)
    # Config is already resolved for all other paths. Serialize only the keys
    # needed by discovery into a standalone config.
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    pairs = discover_task_pairs(config)
    assert len(pairs) == 1
    assert bool(pairs.iloc[0].ready)


def _synthetic_feat_run(root: Path) -> tuple[Path, Path, Path]:
    """One tiny FEAT run on a 2-mm grid shared by function, anatomy and ROIs."""
    rng = np.random.default_rng(0)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    shape, n_volumes = (8, 8, 8), 80
    feat = root / "AP_feat" / "sub-0001_ses-01.feat"
    anat = root / "anat" / "sub-0001_ses-01.anat"
    rois = root / "rois"
    for directory in (feat / "reg", feat / "mc", anat, rois):
        directory.mkdir(parents=True)
    events = pd.DataFrame(
        {
            "onset": np.arange(4.0, 139.0, 9.0),
            "duration": 0.5,
            "trial_type": ["sad", "neutral", "happy"] * 5,
        }
    )
    events_path = root / "events.tsv"
    events.to_csv(events_path, sep="\t", index=False)
    design, names, _ = _design_matrix(
        events, n_volumes, 2.0, np.zeros((n_volumes, 6)), np.zeros((n_volumes, 2))
    )
    bold = 1000.0 + rng.normal(0, 0.05, (*shape, n_volumes))
    roi_masks = {"R1": (slice(1, 3),) * 3, "R2": (slice(5, 7),) * 3}
    for gain, (name, index) in zip((40.0, 10.0), roi_masks.items(), strict=True):
        bold[index] += gain * design[:, 1 + names.index("sad")]
        # different baselines per ROI, as across coils/positions
        bold[index] *= 3.0 if name == "R1" else 0.5
        mask = np.zeros(shape, dtype=np.uint8)
        mask[index] = 1
        nib.save(nib.Nifti1Image(mask, affine), rois / f"{name}.nii")
    nib.save(nib.Nifti1Image(bold.astype(np.float32), affine), feat / "filtered_func_data.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), affine), feat / "reg" / "standard.nii.gz")
    for name in ("example_func2standard.mat", "example_func2highres.mat"):
        np.savetxt(feat / "reg" / name, np.eye(4))
    np.savetxt(feat / "mc" / "prefiltered_func_data_mcf.par", rng.normal(0, 0.01, (n_volumes, 6)))
    (feat / "design.fsf").write_text("set fmri(tr) 2.0\n", encoding="utf-8")
    tissue = np.zeros(shape, np.float32)
    nib.save(nib.Nifti1Image(tissue, affine), anat / "T1_biascorr_brain.nii.gz")
    for name, index in (("T1_fast_pve_0", (0, slice(None), slice(None))), ("T1_fast_pve_2", (7,))):
        probability = np.zeros(shape, np.float32)
        probability[index] = 1.0
        nib.save(nib.Nifti1Image(probability, affine), anat / f"{name}.nii.gz")
    return feat, events_path, rois


def test_percent_signal_change_rescales_task_betas_but_not_residuals(tmp_path: Path) -> None:
    feat, events_path, rois = _synthetic_feat_run(tmp_path)
    spec = {
        "sad_vs_neutral": {
            "positive": "sad",
            "negative": "neutral",
            "rois": ["R1", "R2"],
            "ppi_edges": [["R1", "R2"]],
        }
    }

    def fit(scaling: str) -> dict:
        config = Config(
            tmp_path / "config.yaml",
            {
                "inputs": {"task_data_root": str(tmp_path)},
                "tozzi": {"signal_scaling": scaling, "task_contrasts": {"AP": spec}},
            },
        )
        return _fit_task_run(feat, events_path, ["R1", "R2"], rois, "AP", config)

    raw, psc = fit("raw"), fit("percent_signal_change")
    baselines = {name: raw["roi_series"][name].mean() for name in ("R1", "R2")}
    for name in ("R1", "R2"):
        key = f"AP_sad_vs_neutral_{name}_activation_adapted"
        assert np.isclose(psc["activation"][key], raw["activation"][key] * 100 / baselines[name])
        assert np.allclose(psc["residuals"][name], raw["residuals"][name])
    # baselines differ sixfold, the injected responses fourfold; PSC recovers the latter
    ratio = (
        psc["activation"]["AP_sad_vs_neutral_R1_activation_adapted"]
        / psc["activation"]["AP_sad_vs_neutral_R2_activation_adapted"]
    )
    assert 3.0 < ratio < 5.5
    assert set(psc["ppi"]) == set(raw["ppi"])


def test_feature_catalog_labels_tozzi_definitions_and_duplicates(tmp_path: Path) -> None:
    import yaml

    from for2107_repro.tozzi_pipeline import _feature_selection_table

    mapping = tmp_path / "map.yaml"
    mapping.write_text(
        yaml.safe_dump(
            {
                "tiers": {"t": {"include_tasks": ["rest"]}},
                "features": [
                    {
                        "name": "S1S2_salience_ai_l_ai_r_connectivity",
                        "circuit": "salience",
                        "source_task": "rest",
                        "modality": "connectivity",
                        "equivalence": "published_edge",
                        "rois": ["Salience_AI_L", "Salience_AI_R"],
                    },
                    {
                        "name": "Negativesad_Amy_L__Negativesad_pgACC_M_connectivity",
                        "circuit": "negative_affect_sad",
                        "source_task": "rest",
                        "modality": "connectivity",
                        "equivalence": "within_network_extension",
                        "rois": ["Negativesad_Amy_L", "Negativesad_pgACC_M"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    edge = ["Negativesad_Amy_L", "Negativesad_pgACC_M"]
    config = Config(
        tmp_path / "config.yaml",
        {
            "features": {"mapping": str(mapping), "tier": "t"},
            "tozzi": {
                "task_rois": ["Negativethreat_Amy_L"],
                "task_contrasts": {
                    "AP": {
                        "sad_vs_neutral": {
                            "positive": "sad",
                            "negative": "neutral",
                            "ppi_edges": [edge],
                        }
                    },
                    "FM": {
                        "faces_vs_shapes": {"positive": "F", "negative": "S", "ppi_edges": [edge]}
                    },
                },
                "construct_matches": {
                    "AP_sad_vs_neutral": {"Negativesad": "NS"},
                    "FM_faces_vs_shapes": {"Negativethreat": "NTC"},
                },
            },
        },
    )
    rng = np.random.default_rng(1)
    amygdala, fm_amygdala = rng.normal(size=5), rng.normal(size=5)
    columns = {
        "S1S2_salience_ai_l_ai_r_connectivity__AP": rng.normal(size=5),
        "Negativesad_Amy_L__Negativesad_pgACC_M_connectivity": rng.normal(size=5),
        "AP_sad_vs_neutral_Negativesad_Amy_L_activation_adapted": amygdala,
        "AP_sad_vs_neutral_Negativethreat_Amy_L_activation_adapted": amygdala.copy(),
        "AP_sad_vs_neutral_Salience_AI_L_activation_adapted": rng.normal(size=5),
        "FM_faces_vs_shapes_Negativesad_pgACC_M_activation_adapted": rng.normal(size=5),
        "FM_faces_vs_shapes_Negativesad_Amy_L_activation_adapted": fm_amygdala,
        "FM_faces_vs_shapes_Negativethreat_Amy_L_activation_adapted": fm_amygdala.copy(),
        "AP_sad_vs_neutral_Negativesad_Amy_L__Negativesad_pgACC_M_ppi_adapted"
        "__phys_Negativesad_pgACC_M": rng.normal(size=5),
        "FM_faces_vs_shapes_Negativesad_Amy_L__Negativesad_pgACC_M_ppi_adapted": rng.normal(size=5),
    }
    features = pd.DataFrame({"subject_id": [f"s{i}" for i in range(5)], **columns})
    catalog = _feature_selection_table(config, features).set_index("feature")
    assert set(catalog.index) == set(columns)
    labels = catalog[["equivalence", "tozzi_circuit"]].agg("|".join, axis=1).to_dict()
    assert labels == {
        "S1S2_salience_ai_l_ai_r_connectivity__AP": "published_edge|S_C",
        "Negativesad_Amy_L__Negativesad_pgACC_M_connectivity": "within_network_extension|",
        "AP_sad_vs_neutral_Negativesad_Amy_L_activation_adapted": "closest_task_analogue|NS_A",
        "AP_sad_vs_neutral_Negativethreat_Amy_L_activation_adapted": "tozzi_rois_other_contrast|",
        "AP_sad_vs_neutral_Salience_AI_L_activation_adapted": "network_extension|",
        "FM_faces_vs_shapes_Negativesad_pgACC_M_activation_adapted": "tozzi_rois_other_contrast|",
        "FM_faces_vs_shapes_Negativesad_Amy_L_activation_adapted": "tozzi_rois_other_contrast|",
        "FM_faces_vs_shapes_Negativethreat_Amy_L_activation_adapted": "closest_task_analogue|NTC_A",
        "AP_sad_vs_neutral_Negativesad_Amy_L__Negativesad_pgACC_M_ppi_adapted"
        "__phys_Negativesad_pgACC_M": "closest_task_analogue|NS_C",
        "FM_faces_vs_shapes_Negativesad_Amy_L__Negativesad_pgACC_M_ppi_adapted": (
            "tozzi_rois_other_contrast|"
        ),
    }
    assert catalog.loc["S1S2_salience_ai_l_ai_r_connectivity__AP", "source_task"] == "AP"
    assert (
        catalog.loc["AP_sad_vs_neutral_Negativethreat_Amy_L_activation_adapted", "duplicate_of"]
        == "AP_sad_vs_neutral_Negativesad_Amy_L_activation_adapted"
    )
    # the copy that is a Tozzi feature is kept, even when it sorts later
    assert (
        catalog.loc["FM_faces_vs_shapes_Negativesad_Amy_L_activation_adapted", "duplicate_of"]
        == "FM_faces_vs_shapes_Negativethreat_Amy_L_activation_adapted"
    )
    assert (catalog.duplicate_of != "").sum() == 2
