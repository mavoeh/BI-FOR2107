import json

from for2107_repro.bids import sanitized_metadata
from for2107_repro.config import load_config
from for2107_repro.inventory import inventory


def test_smoke_inventory() -> None:
    result = inventory(load_config("configs/smoke.yaml"))
    assert result["summary"]["subject_count"] == 10
    assert result["summary"]["task_counts"] == {"AP": 10, "ENC": 10, "FM": 10, "rest": 10}
    assert result["summary"]["missing_task_events"] == 30


def test_sensitive_json_fields_are_removed(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"PatientID": "secret", "RepetitionTime": 2}), encoding="utf-8")
    assert sanitized_metadata(source) == {"RepetitionTime": 2}

