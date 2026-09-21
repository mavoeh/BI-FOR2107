import pandas as pd

from for2107_repro.analysis import run_primary_analysis
from for2107_repro.config import Config, load_config


def test_analysis_is_gated_without_features(tmp_path) -> None:
    original = load_config("configs/smoke.yaml")
    data = {**original.data, "outputs": {**original.section("outputs"), "root": str(tmp_path)}}
    config = Config(original.path, data)
    status = run_primary_analysis(config, pd.DataFrame(), pd.DataFrame())
    assert status == {"ready": False, "reasons": ["no-extracted-features"]}
    assert (tmp_path / "analysis" / "status.json").exists()


def test_analysis_retains_motion_unassessed_when_configured(tmp_path) -> None:
    original = load_config("configs/external_rs.yaml")
    data = {
        **original.data,
        "outputs": {**original.section("outputs"), "root": str(tmp_path)},
    }
    config = Config(original.path, data)
    features = pd.DataFrame(
        {
            "subject_id": ["sub-0001"],
            "feature_a": [0.1],
            "feature_b": [0.2],
            "qc_pass": [float("nan")],
            "motion_qc_available": [False],
        }
    )
    participants = pd.DataFrame(
        {"subject_id": ["sub-0001"], "analysis_role": ["clinical"]}
    )

    status = run_primary_analysis(config, features, participants)

    assert status["clinical_n"] == 1
    assert status["healthy_n"] == 0
