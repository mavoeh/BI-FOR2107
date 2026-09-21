
from for2107_repro.config import load_config


def test_smoke_paths_are_resolved() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.require_path("inputs.bids_root").is_absolute()
    assert config.require_path("inputs.bids_root").name == "raw_examples"
    assert config.get("selection.session") == "ses-01"


def test_external_rest_path_is_resolved() -> None:
    config = load_config("configs/external_rs.yaml")
    assert config.require_path("inputs.preprocessed_rest_root").is_absolute()
    assert config.get("preprocessing.missing_motion_policy") == "include_unassessed"
